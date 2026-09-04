/*
 * tool_tracker_onnx.cpp
 * FOD Tool Tracking System — ONNX Runtime + OpenCV + ByteTrack
 *
 * Classes : 0 drill | 1 hammer | 2 pliers | 3 screwdriver | 4 wrench
 * Input   : 512×512 ONNX model (best.onnx)
 *
 * Track lifecycle
 *   PENDING  → visible for CONFIRM_FRAMES consecutive frames → ACTIVE  (log DETECTED)
 *   ACTIVE   → absent ≥ 1 frame                              → MISSING (record timestamp)
 *   MISSING  → visible again                                 → ACTIVE
 *   MISSING  → absent ≥ ABSENT_SEC seconds                   → REMOVED (log REMOVED)
 *   MISSING  → absent ≥ ALERT_SEC seconds                    → log ALERT (once)
 *
 * Controls : q / ESC = quit   (writes session_summary.txt on exit)
 *
 * Portable: pure C++17 + ONNX Runtime C++ API + OpenCV.
 * Swap ORT provider (CPU → TensorRT) for Jetson Orin Nano Super deployment.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
// ADD for TensorRT
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>
//take out for TensorRT
//#include <onnxruntime_cxx_api.h>

#include "bytetrack.h"

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int   INPUT_W         = 512;
static constexpr int   INPUT_H         = 512;
static constexpr float CONF_THRESHOLD  = 0.35f;  // threshold passed to ByteTrack, increased a little to filter out weak ghost detections
static constexpr float NMS_IOU         = 0.3f;   //accounts for how much overlap between bounding boxes before a tool disappears behind another
static constexpr int   CONFIRM_FRAMES  = 5;      // consecutive visible frames before DETECTED
static constexpr double ABSENT_SEC     = 5.0;      // consecutive absent seconds before REMOVED
static constexpr double ALERT_SEC      = 60.0;    // seconds absent before ALERT

static const std::string MODEL_PATH    = "best.onnx";
static const std::string LOG_FILE      = "tool_log.txt";
static const std::string SUMMARY_FILE  = "session_summary.txt";
static const std::string WINDOW_TITLE  = "FOD Tool Tracker [ByteTrack + ONNX Runtime]";

static const std::array<std::string, 5> CLASS_NAMES = {
    "drill", "hammer", "pliers", "screwdriver", "wrench"
};

static const std::array<cv::Scalar, 5> CLASS_COLOURS = {{
    {  0, 165, 255},   // drill        – orange
    {  0, 255,   0},   // hammer       – green
    {255,  50,  50},   // pliers       – blue
    {  0,   0, 255},   // screwdriver  – red
    {255,   0, 255},   // wrench       – magenta
}};

// ─────────────────────────────────────────────────────────────────────────────
// Time helpers
// ─────────────────────────────────────────────────────────────────────────────

using Clock = std::chrono::steady_clock;
using TP    = std::chrono::steady_clock::time_point;

static std::string wall_timestamp() {
    auto t  = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(t);
    std::tm tm_buf{}; localtime_r(&tt, &tm_buf);
    std::ostringstream ss;
    ss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

static double elapsed_sec(TP from, TP to = Clock::now()) {
    return std::chrono::duration<double>(to - from).count();
}

static std::string fmt_duration(double sec) {
    if(sec < 60)  return std::to_string((int)sec) + "s";
    int m = (int)(sec/60), s = (int)sec%60;
    return std::to_string(m) + "m" + std::to_string(s) + "s";
}

// ─────────────────────────────────────────────────────────────────────────────
// Logging
// ─────────────────────────────────────────────────────────────────────────────

static std::ofstream g_log;

static void log_line(const std::string& line) {
    std::cout << line << "\n";
    if(g_log.is_open()) { g_log << line << "\n"; g_log.flush(); }
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolRecord — per-track state machine
// ─────────────────────────────────────────────────────────────────────────────

enum class ToolState { PENDING, ACTIVE, MISSING, REMOVED };

struct ToolRecord {
    int         track_id;
    int         cls;
    std::string name;
    float       conf        = 0.f;
    float       x1=0,y1=0,x2=0,y2=0;   // last known box

    ToolState   state       = ToolState::PENDING;
    int         confirm_frames = 0;  // consecutive visible
    int         absent_frames  = 0;  // consecutive absent

    TP  first_seen_at;
    TP  confirmed_at;
    TP  last_seen_at;
    TP  went_missing_at;

    bool alert_fired = false;

    double seconds_present() const {
        if(state == ToolState::ACTIVE)
            return elapsed_sec(confirmed_at);
        if(state == ToolState::MISSING || state == ToolState::REMOVED)
            return elapsed_sec(confirmed_at, went_missing_at);
        return 0.0;
    }

    double seconds_absent() const {
        if(state == ToolState::MISSING)
            return elapsed_sec(went_missing_at);
        return 0.0;
    }

    bool alert_due() const {
        return state == ToolState::MISSING
            && !alert_fired
            && seconds_absent() >= ALERT_SEC;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Preprocessing — letterbox resize + BGR→RGB→float CHW blob
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<float> preprocess(const cv::Mat& frame,
                                      float& scale, int& pad_l, int& pad_t) {
    float r  = std::min((float)INPUT_W/frame.cols, (float)INPUT_H/frame.rows);
    scale    = r;
    int nw   = (int)std::round(frame.cols*r), nh = (int)std::round(frame.rows*r);
    pad_l    = (INPUT_W-nw)/2;
    pad_t    = (INPUT_H-nh)/2;

    cv::Mat resized; cv::resize(frame, resized, cv::Size(nw, nh));
    cv::Mat canvas(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114,114,114));
    resized.copyTo(canvas(cv::Rect(pad_l, pad_t, nw, nh)));

    cv::Mat rgb; cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.f/255.f);

    std::vector<cv::Mat> ch(3); cv::split(rgb, ch);
    std::vector<float> blob; blob.reserve(3*INPUT_H*INPUT_W);
    for(int c=0;c<3;++c){
        auto* p = (const float*)ch[c].data;
        blob.insert(blob.end(), p, p+INPUT_H*INPUT_W);
    }
    return blob;
}

// ─────────────────────────────────────────────────────────────────────────────
// Post-processing — decode YOLO head + per-class NMS → TBox vector
// YOLO output layout: [1, 4+nc, num_anchors]  (coordinates in INPUT space)
// ─────────────────────────────────────────────────────────────────────────────

static float det_iou(const TBox& a, const TBox& b) {
    float ix1=std::max(a.x1,b.x1), iy1=std::max(a.y1,b.y1);
    float ix2=std::min(a.x2,b.x2), iy2=std::min(a.y2,b.y2);
    float inter=std::max(0.f,ix2-ix1)*std::max(0.f,iy2-iy1);
    float ua=(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-inter+1e-6f;
    return inter/ua;
}

static std::vector<TBox> postprocess(const float* data,
                                      int n_anchors, int n_classes,
                                      float scale, int pad_l, int pad_t,
                                      int orig_w,  int orig_h) {
    // Decode
    std::vector<TBox> raw;
    for(int a=0;a<n_anchors;++a){
        float cx=data[0*n_anchors+a], cy=data[1*n_anchors+a];
        float bw=data[2*n_anchors+a], bh=data[3*n_anchors+a];
        float best=0.f; int bc=-1;
        for(int c=0;c<n_classes;++c){
            float s=data[(4+c)*n_anchors+a];
            if(s>best){ best=s; bc=c; }
        }
        if(best < CONF_THRESHOLD) continue;
        float x1=std::max(0.f,(cx-bw*.5f-pad_l)/scale);
        float y1=std::max(0.f,(cy-bh*.5f-pad_t)/scale);
        float x2=std::min((float)orig_w,(cx+bw*.5f-pad_l)/scale);
        float y2=std::min((float)orig_h,(cy+bh*.5f-pad_t)/scale);
        raw.push_back({x1,y1,x2,y2,best,bc});
    }
    // Per-class NMS
    std::sort(raw.begin(),raw.end(),[](const TBox& a,const TBox& b){ return a.score>b.score; });
    std::vector<bool> sup(raw.size(),false);
    std::vector<TBox> out;
    for(size_t i=0;i<raw.size();++i){
        if(sup[i]) continue;
        out.push_back(raw[i]);
        for(size_t j=i+1;j<raw.size();++j)
            if(!sup[j] && raw[i].cls==raw[j].cls && det_iou(raw[i],raw[j])>NMS_IOU)
                sup[j]=true;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// HUD drawing
// ─────────────────────────────────────────────────────────────────────────────

static cv::Scalar class_colour(int cls) {
    return CLASS_COLOURS[cls % CLASS_COLOURS.size()];
}

static void draw_dashed_rect(cv::Mat& img,
                              cv::Rect rect,
                              cv::Scalar col,
                              int thickness = 1,
                              int dash = 10) {
    // Simulate a dashed rectangle for MISSING tool ghost boxes
    auto draw_dashed = [&](cv::Point p1, cv::Point p2) {
        float len = std::sqrt((float)(p2.x-p1.x)*(p2.x-p1.x)+(p2.y-p1.y)*(p2.y-p1.y));
        if(len < 1) return;
        float dx=(p2.x-p1.x)/len, dy=(p2.y-p1.y)/len;
        float pos=0;
        bool  draw_seg=true;
        while(pos<len){
            float end_pos=std::min(pos+(float)dash, len);
            if(draw_seg)
                cv::line(img, {p1.x+(int)(pos*dx), p1.y+(int)(pos*dy)},
                              {p1.x+(int)(end_pos*dx), p1.y+(int)(end_pos*dy)}, col, thickness);
            pos=end_pos; draw_seg=!draw_seg;
        }
    };
    cv::Point tl=rect.tl(), tr={rect.x+rect.width, rect.y};
    cv::Point bl={rect.x,   rect.y+rect.height}, br=rect.br();
    draw_dashed(tl,tr); draw_dashed(tr,br);
    draw_dashed(br,bl); draw_dashed(bl,tl);
}

static void draw_hud(cv::Mat& frame,
                     const std::map<int, ToolRecord>& records,
                     double fps, int frame_idx) {
    const int W = frame.cols, H = frame.rows;

    // ── Bounding boxes for ACTIVE / PENDING tracks ────────────────────────
    for(const auto& [id, rec] : records) {
        if(rec.state == ToolState::REMOVED) continue;

        cv::Rect box((int)rec.x1,(int)rec.y1,
                     (int)(rec.x2-rec.x1),(int)(rec.y2-rec.y1));
        box &= cv::Rect(0,0,W,H);  // clamp to frame
        if(box.empty()) continue;

        if(rec.state == ToolState::ACTIVE) {
            cv::Scalar col = class_colour(rec.cls);
            cv::rectangle(frame, box, col, 2);

            // Label: "drill #3 | 0.91 | 12s"
            char buf[64];
            std::snprintf(buf,sizeof(buf),"%s #%d | %.0f%% | %s",
                rec.name.c_str(), rec.track_id,
                rec.conf*100, fmt_duration(rec.seconds_present()).c_str());

            int baseline=0;
            cv::Size ts = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.48, 1, &baseline);
            int ty = std::max((int)rec.y1 - 6, ts.height+4);
            cv::rectangle(frame,{(int)rec.x1, ty-ts.height-2},
                                 {(int)rec.x1+ts.width+4, ty+baseline}, col, cv::FILLED);
            cv::putText(frame, buf, {(int)rec.x1+2, ty},
                        cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0,0,0), 1, cv::LINE_AA);

        } else if(rec.state == ToolState::PENDING) {
            // Thin yellow box while accumulating confirmation frames, commented out so we have a cleaner demo and no ghost boxes
            //draw_dashed_rect(frame, box, cv::Scalar(0,220,220), 1);

        } else if(rec.state == ToolState::MISSING) {
            // Ghost box at last known position
            bool alerting = rec.alert_fired || rec.alert_due();
            cv::Scalar col = alerting ? cv::Scalar(0,0,255) : cv::Scalar(0,140,255);
            draw_dashed_rect(frame, box, col, 2);
            char buf[48];
            std::snprintf(buf,sizeof(buf),"MISSING #%d %s",
                rec.track_id, fmt_duration(rec.seconds_absent()).c_str());
            cv::putText(frame, buf, {(int)rec.x1, std::max((int)rec.y1-6,12)},
                        cv::FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv::LINE_AA);
        }
    }

    // ── Status panel (top-right) ──────────────────────────────────────────
    const int PW = 280, LHGT = 20, PAD = 6;
    const int MAX_PANEL_ENTRIES = 10;

    std::vector<const ToolRecord*> panel_list;
    for(const auto& [id,rec] : records)
        if(rec.state != ToolState::REMOVED) panel_list.push_back(&rec);

    std::sort(panel_list.begin(), panel_list.end(),
        [](const ToolRecord* a, const ToolRecord* b){ return a->last_seen_at > b->last_seen_at; });

    if((int)panel_list.size() > MAX_PANEL_ENTRIES)
        panel_list.resize(MAX_PANEL_ENTRIES);

    int panel_entries = (int)panel_list.size();
    int ph = PAD + panel_entries * LHGT + PAD;
    int px = W - PW - 4, py = 4;
    cv::rectangle(frame,{px,py},{px+PW,py+ph},{30,30,30},-1);
    cv::rectangle(frame,{px,py},{px+PW,py+ph},{100,100,100},1);

    int ry = py + PAD + LHGT - 4;
    for(const auto* rec_ptr : panel_list) {
        const ToolRecord& rec = *rec_ptr;
        char buf[80];
        cv::Scalar col;
        if(rec.state == ToolState::ACTIVE) {
            col = class_colour(rec.cls);
            std::snprintf(buf,sizeof(buf),"[OK]  %s #%d  %.0f%%  %s",
                rec.name.c_str(), rec.track_id, rec.conf*100,
                fmt_duration(rec.seconds_present()).c_str());
        } else if(rec.state == ToolState::PENDING) {
            col = cv::Scalar(0,220,220);
            std::snprintf(buf,sizeof(buf),"[..] %s #%d  (%d/10)",
                rec.name.c_str(), rec.track_id, rec.confirm_frames);
        } else { // MISSING
            bool alerted = rec.alert_fired || rec.alert_due();
            col = alerted ? cv::Scalar(0,80,255) : cv::Scalar(0,165,255);
            std::snprintf(buf,sizeof(buf),"%s %s #%d  absent:%s",
                alerted?"[!!]":"[!] ",
                rec.name.c_str(), rec.track_id,
                fmt_duration(rec.seconds_absent()).c_str());
        }
        cv::putText(frame, buf, {px+PAD, ry},
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv::LINE_AA);
        ry += LHGT;
    }

    // ── Top bar ───────────────────────────────────────────────────────────
    char topbar[120];
    std::snprintf(topbar,sizeof(topbar),
        "FOD Tool Tracker | Frame:%d | FPS:%.1f", frame_idx, fps);
    cv::rectangle(frame,{0,0},{W,28},{20,20,20},-1);
    cv::putText(frame,topbar,{8,20},
                cv::FONT_HERSHEY_SIMPLEX,0.60,{220,220,220},1,cv::LINE_AA);

    // ── Active ALERTs – red banner at bottom ─────────────────────────────
    int ay = H - 10;
    for(const auto& [id, rec] : records) {
        if(!(rec.state == ToolState::MISSING && rec.alert_fired)) continue;
        char abuf[80];
        std::snprintf(abuf,sizeof(abuf),
            "!! ALERT: %s #%d missing %.0fs !!",
            rec.name.c_str(), rec.track_id, rec.seconds_absent());
        cv::putText(frame, abuf, {10, ay},
                    cv::FONT_HERSHEY_SIMPLEX, 0.70, cv::Scalar(0,0,255), 2, cv::LINE_AA);
        ay -= 28;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Session summary
// ─────────────────────────────────────────────────────────────────────────────

static void write_summary(const std::map<int,ToolRecord>& active_records,
                           const std::vector<ToolRecord>&  history,
                           TP session_start) {
    std::ofstream f(SUMMARY_FILE);
    if(!f) { std::cerr << "Could not write " << SUMMARY_FILE << "\n"; return; }

    double dur = elapsed_sec(session_start);
    f << "=== FOD Tool Tracker — Session Summary ===\n";
    f << "Date     : " << wall_timestamp() << "\n";
    f << "Duration : " << fmt_duration(dur) << "\n\n";

    // Stats
    int total_det=0, total_rem=0;
    for(const auto& r : history){ total_det++; if(r.state==ToolState::REMOVED) total_rem++; }
    for(const auto& [id,r] : active_records) if(r.state==ToolState::ACTIVE||r.state==ToolState::MISSING) total_det++;

    f << "Total detections : " << total_det << "\n";
    f << "Total removals   : " << total_rem << "\n\n";

    // Active tools at session end
    f << "--- Tools present at session end ---\n";
    bool any_active=false;
    for(const auto& [id,r] : active_records) if(r.state==ToolState::ACTIVE){
        f << "  " << r.name << " (track #" << r.track_id << ")  present for "
          << fmt_duration(r.seconds_present()) << "\n";
        any_active=true;
    }
    if(!any_active) f << "  (none)\n";
    f << "\n";

    // Unreturned tools — FOD risk
    f << "--- Unreturned tools (FOD risk) ---\n";
    bool any_fod=false;
    for(const auto& [id,r] : active_records) if(r.state==ToolState::MISSING){
        f << "  !! " << r.name << " (track #" << r.track_id
          << ")  MISSING for " << fmt_duration(r.seconds_absent()) << "\n";
        any_fod=true;
    }
    for(const auto& r : history) if(r.state==ToolState::REMOVED){
        f << "  -- " << r.name << " (track #" << r.track_id
          << ")  removed after " << fmt_duration(r.seconds_present()) << "\n";
        any_fod=true;
    }
    if(!any_fod) f << "  (none — all tools accounted for)\n";
    f << "\n";

    // Detection history
    f << "--- Detection log ---\n";
    for(const auto& r : history) {
        f << "  " << r.name << " #" << r.track_id
          << "  present:" << fmt_duration(r.seconds_present())
          << "  state:" << (r.state==ToolState::REMOVED?"REMOVED":"OTHER") << "\n";
    }
    f.close();
    std::cout << "Session summary written to " << SUMMARY_FILE << "\n";
}

class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << msg << "\n";
    }
} trt_logger;


// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    // ── ONNX Runtime ─────────────────────────────────────────────────────
    /* Pre TensorRT - CPU
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "tool_tracker");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    std::cout << "Loading " << MODEL_PATH << " ...\n";
    Ort::Session session(env, MODEL_PATH.c_str(), opts);
    */
    // 01 TensorRT Load Engine implementation 
    std::ifstream engineFile("best.engine", std::ios::binary);
    std::vector<char> engineData((std::istreambuf_iterator<char>(engineFile)), std::istreambuf_iterator<char>());
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(trt_logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    std::string in_name, out_name;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) in_name = name;
        else out_name = name;
    }

    nvinfer1::Dims out_dims_static = engine->getTensorShape(out_name.c_str());
    int n_classes = (out_dims_static.nbDims >= 2 && out_dims_static.d[1] > 4)
                  ? out_dims_static.d[1] - 4 : (int)CLASS_NAMES.size();
    int n_anchors = (out_dims_static.nbDims >= 3) ? out_dims_static.d[2] : 0;

    const size_t in_numel = 3 * INPUT_H * INPUT_W;
    size_t out_numel = 1;
    for (int d = 0; d < out_dims_static.nbDims; ++d) out_numel *= out_dims_static.d[d];

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ── Log file ─────────────────────────────────────────────────────────
    g_log.open(LOG_FILE, std::ios::app);
    log_line("\n=== FOD Tool Tracker session started " + wall_timestamp() + " ===");

    // ── ByteTracker ───────────────────────────────────────────────────────
    // high_thresh=0.5 splits detections; iou thresholds tuned for static tools
    BYTETracker tracker(/*high_thresh=*/0.50f,
                        /*iou_t1=*/0.80f, /*iou_t2=*/0.50f, /*iou_t3=*/0.70f,
                        /*max_lost=*/30);

    // ── Webcam ───────────────────────────────────────────────────────────
    cv::VideoCapture cap(0);
    if(!cap.isOpened()){
        std::cerr << "ERROR: Cannot open webcam 0.\n"; return 1;
    }

    std::cout << "Warming up camera ...\n";
    for(int i=0;i<10;++i){ cv::Mat t; cap.read(t); if(!t.empty()) break; cv::waitKey(100); }

    cv::Mat test; cap.read(test);
    if(test.empty()){ std::cerr << "ERROR: Camera returned no frames.\n"; return 1; }
    std::cout << "Camera: " << test.cols << "×" << test.rows << "\n\n";
    std::cout << "Controls: [q/ESC] quit\n\n";

    cv::namedWindow(WINDOW_TITLE, cv::WINDOW_NORMAL);

    // ── Per-track records + history ───────────────────────────────────────
    std::map<int, ToolRecord> records;   // keyed by ByteTrack track_id
    std::vector<ToolRecord>   history;   // REMOVED records for session summary

    TP session_start = Clock::now();
    int frame_idx = 0;

    // Rolling FPS
    std::vector<double> fps_buf;
    fps_buf.reserve(30);

    //02 TensorRT allocate buffers
    void* buffers[2];
    cudaMalloc(&buffers[0], in_numel * sizeof(float));
    cudaMalloc(&buffers[1], out_numel * sizeof(float));

    context->setTensorAddress(in_name.c_str(), buffers[0]);
    context->setTensorAddress(out_name.c_str(), buffers[1]);

    // ── Main loop ─────────────────────────────────────────────────────────
    //Loop through every frame
    while(true) {
        cv::Mat frame; cap.read(frame);
        if(frame.empty()) { std::cerr << "Warning: empty frame.\n"; continue; }
        ++frame_idx;
        auto frame_start = Clock::now();

        // ── Inference ────────────────────────────────────────────────────
        float scale; int pad_l, pad_t;
        auto blob = preprocess(frame, scale, pad_l, pad_t);

        /*old cpu copy in copy out
        Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            mi, blob.data(), in_numel, in_shape.data(), in_shape.size());

        auto outputs = session.Run(Ort::RunOptions{nullptr},
                                   in_names, &in_tensor, 1, out_names, 1);

        const float* out_data = outputs[0].GetTensorData<float>();
        */

        //03-05 TensorRT copy into GPU, run, copy out happens every single frame
        cudaMemcpyAsync(buffers[0], blob.data(), in_numel * sizeof(float), cudaMemcpyHostToDevice, stream);
        context->enqueueV3(stream);

        std::vector<float> out_data(out_numel);
        cudaMemcpyAsync(out_data.data(), buffers[1], out_numel * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::vector<TBox> dets = postprocess(out_data.data(), n_anchors, n_classes,
                                              scale, pad_l, pad_t,
                                              frame.cols, frame.rows);

        // ── ByteTrack update ─────────────────────────────────────────────
        std::vector<TrackedBox> active_tracks = tracker.update(dets);

        // Which track IDs are visible this frame
        std::map<int, TrackedBox> visible;
        for(const auto& tb : active_tracks) visible[tb.track_id] = tb;

        TP now = Clock::now();

        // ── Update records for tracks that ARE visible ────────────────────
        for(const auto& [tid, tb] : visible) {
            auto it = records.find(tid);
            if(it == records.end()) {
                // New track — create PENDING record
                ToolRecord r;
                r.track_id      = tid;
                r.cls           = tb.cls;
                r.name          = (tb.cls < (int)CLASS_NAMES.size())
                                  ? CLASS_NAMES[tb.cls] : "cls"+std::to_string(tb.cls);
                r.first_seen_at = now;
                records[tid]    = r;
                it              = records.find(tid);
            }
            ToolRecord& r = it->second;
            r.conf         = tb.score;
            r.x1=tb.x1; r.y1=tb.y1; r.x2=tb.x2; r.y2=tb.y2;
            r.last_seen_at = now;
            r.absent_frames = 0;

            if(r.state == ToolState::PENDING) {
                ++r.confirm_frames;
                if(r.confirm_frames >= CONFIRM_FRAMES) {
                    r.state        = ToolState::ACTIVE;
                    r.confirmed_at = now;
                    char buf[120];
                    std::snprintf(buf,sizeof(buf),
                        "[%s] DETECTED: %s (track #%d)  conf:%.2f",
                        wall_timestamp().c_str(), r.name.c_str(), r.track_id, r.conf);
                    log_line(buf);
                }
            } else if(r.state == ToolState::MISSING) {
                r.state = ToolState::ACTIVE;
                r.alert_fired = false;  // reset so we can alert again if it goes missing
            }
        }

        // ── Update records for tracks that are NOT visible ────────────────
        for(auto& [tid, r] : records) {
            if(visible.count(tid)) continue;                        // handled above
            if(r.state == ToolState::REMOVED) continue;

            if(r.state == ToolState::PENDING) {
                r.confirm_frames = 0;                               // reset counter
                continue;
            }

            ++r.absent_frames;

            if(r.state == ToolState::ACTIVE && r.absent_frames == 1) {
                r.state          = ToolState::MISSING;
                r.went_missing_at = now;
            }

            if(r.state == ToolState::MISSING) {
                // Alert check (≥60s absent, fire once)
                if(r.alert_due()) {
                    r.alert_fired = true;
                    char buf[120];
                    std::snprintf(buf,sizeof(buf),
                        "[%s] ALERT: %s (track #%d) absent for %.0fs",
                        wall_timestamp().c_str(), r.name.c_str(),
                        r.track_id, r.seconds_absent());
                    log_line(buf);
                }

                // Removal after ABSENT_SEC seconds missing
                if(r.seconds_absent() >= ABSENT_SEC) {
                    r.state = ToolState::REMOVED;
                    char buf[140];
                    std::snprintf(buf,sizeof(buf),
                        "[%s] REMOVED: %s (track #%d)  was present for %s",
                        wall_timestamp().c_str(), r.name.c_str(),
                        r.track_id, fmt_duration(r.seconds_present()).c_str());
                    log_line(buf);
                    history.push_back(r);
                }
            }
        }

        // ── FPS ──────────────────────────────────────────────────────────
        double frame_ms = elapsed_sec(frame_start)*1000.0;
        fps_buf.push_back(frame_ms);
        if((int)fps_buf.size() > 30) fps_buf.erase(fps_buf.begin());
        double avg_ms = std::accumulate(fps_buf.begin(),fps_buf.end(),0.0)/fps_buf.size();
        double fps    = 1000.0/avg_ms;

        // ── Draw + display ────────────────────────────────────────────────
        draw_hud(frame, records, fps, frame_idx);
        cv::imshow(WINDOW_TITLE, frame);

        int key = cv::waitKey(1) & 0xFF;
        if(key == 'q' || key == 27) break;
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    cap.release();
    cv::destroyAllWindows();

    log_line("=== FOD Tool Tracker session ended " + wall_timestamp() + " ===\n");
    g_log.close();

    write_summary(records, history, session_start);
    return 0;
}
