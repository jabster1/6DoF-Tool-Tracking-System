/*
 * bytetrack.h
 * Self-contained ByteTrack implementation — no external dependencies.
 *
 * Kalman state : [cx, cy, w, h, vx, vy, vw, vh]
 * Input        : TBox   {x1,y1,x2,y2, score, cls}
 * Output       : TrackedBox {x1,y1,x2,y2, score, cls, track_id}
 *
 * Two-stage IoU association:
 *   Stage 1 – high-conf dets  vs active  tracks
 *   Stage 2 – low-conf dets   vs unmatched active tracks
 *   Stage 3 – remaining dets  vs lost (buffered) tracks → re-activate
 *             remaining dets  with no match           → new tracks
 *
 * Compatible with C++17.  Mark every free function inline to keep this
 * safely includable from multiple translation units.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

struct TBox {
    float x1, y1, x2, y2, score;
    int   cls;
};

struct TrackedBox {
    float x1, y1, x2, y2, score;
    int   cls, track_id;
};

// ─────────────────────────────────────────────────────────────────────────────
// IoU
// ─────────────────────────────────────────────────────────────────────────────

inline float tbox_iou(const TBox& a, const TBox& b) {
    float ix1  = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2  = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, ix2-ix1) * std::max(0.f, iy2-iy1);
    float ua    = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1)
                  - inter + 1e-6f;
    return inter / ua;
}

// ─────────────────────────────────────────────────────────────────────────────
// Flat-array matrix helpers  (row-major, M[i*COLS + j])
// Sizes used by the Kalman filter:
//   M88 – 8×8   M48 – 4×8   M84 – 8×4   M44 – 4×4
// ─────────────────────────────────────────────────────────────────────────────

using V8  = std::array<float, 8>;
using V4  = std::array<float, 4>;
using M88 = std::array<float, 64>;   // 8×8
using M48 = std::array<float, 32>;   // 4×8
using M84 = std::array<float, 32>;   // 8×4
using M44 = std::array<float, 16>;   // 4×4

// matrix-vector products
inline V8 mv_88_8 (const M88& A, const V8& x){ V8 y{}; for(int i=0;i<8;++i) for(int j=0;j<8;++j) y[i]+=A[i*8+j]*x[j]; return y; }
inline V4 mv_48_8 (const M48& A, const V8& x){ V4 y{}; for(int i=0;i<4;++i) for(int j=0;j<8;++j) y[i]+=A[i*8+j]*x[j]; return y; }
inline V8 mv_84_4 (const M84& A, const V4& x){ V8 y{}; for(int i=0;i<8;++i) for(int j=0;j<4;++j) y[i]+=A[i*4+j]*x[j]; return y; }
inline V4 mv_44_4 (const M44& A, const V4& x){ V4 y{}; for(int i=0;i<4;++i) for(int j=0;j<4;++j) y[i]+=A[i*4+j]*x[j]; return y; }

// matrix-matrix products
inline M88 mm_88_88(const M88& A,const M88& B){ M88 C{}; for(int i=0;i<8;++i) for(int k=0;k<8;++k) for(int j=0;j<8;++j) C[i*8+j]+=A[i*8+k]*B[k*8+j]; return C; }
inline M48 mm_48_88(const M48& A,const M88& B){ M48 C{}; for(int i=0;i<4;++i) for(int k=0;k<8;++k) for(int j=0;j<8;++j) C[i*8+j]+=A[i*8+k]*B[k*8+j]; return C; }
inline M44 mm_48_84(const M48& A,const M84& B){ M44 C{}; for(int i=0;i<4;++i) for(int k=0;k<8;++k) for(int j=0;j<4;++j) C[i*4+j]+=A[i*8+k]*B[k*4+j]; return C; }
inline M84 mm_88_84(const M88& A,const M84& B){ M84 C{}; for(int i=0;i<8;++i) for(int k=0;k<8;++k) for(int j=0;j<4;++j) C[i*4+j]+=A[i*8+k]*B[k*4+j]; return C; }
inline M84 mm_84_44(const M84& A,const M44& B){ M84 C{}; for(int i=0;i<8;++i) for(int k=0;k<4;++k) for(int j=0;j<4;++j) C[i*4+j]+=A[i*4+k]*B[k*4+j]; return C; }
inline M88 mm_84_48(const M84& A,const M48& B){ M88 C{}; for(int i=0;i<8;++i) for(int k=0;k<4;++k) for(int j=0;j<8;++j) C[i*8+j]+=A[i*4+k]*B[k*8+j]; return C; }

// transposes
inline M88 T_88(const M88& A){ M88 B{}; for(int i=0;i<8;++i) for(int j=0;j<8;++j) B[i*8+j]=A[j*8+i]; return B; }
inline M84 T_48(const M48& A){ M84 B{}; for(int i=0;i<4;++i) for(int j=0;j<8;++j) B[j*4+i]=A[i*8+j]; return B; }

// element-wise
inline M88 add_88(const M88& A,const M88& B){ M88 C; for(int i=0;i<64;++i) C[i]=A[i]+B[i]; return C; }
inline M88 sub_88(const M88& A,const M88& B){ M88 C; for(int i=0;i<64;++i) C[i]=A[i]-B[i]; return C; }
inline M44 add_44(const M44& A,const M44& B){ M44 C; for(int i=0;i<16;++i) C[i]=A[i]+B[i]; return C; }

inline M88 eye_88(){ M88 I{}; for(int i=0;i<8;++i) I[i*8+i]=1.f; return I; }

// 4×4 inverse via Gaussian elimination with partial pivoting
inline M44 inv_44(const M44& M) {
    float a[4][8];
    for(int i=0;i<4;++i){
        for(int j=0;j<4;++j) a[i][j]=M[i*4+j];
        for(int j=0;j<4;++j) a[i][4+j]=(i==j)?1.f:0.f;
    }
    for(int c=0;c<4;++c){
        int piv=c;
        for(int r=c+1;r<4;++r) if(std::abs(a[r][c])>std::abs(a[piv][c])) piv=r;
        for(int j=0;j<8;++j) std::swap(a[c][j],a[piv][j]);
        float d=a[c][c];
        if(std::abs(d)<1e-10f){ M44 I{}; for(int i=0;i<4;++i)I[i*4+i]=1.f; return I; }
        for(int j=0;j<8;++j) a[c][j]/=d;
        for(int r=0;r<4;++r){ if(r==c) continue; float f=a[r][c]; for(int j=0;j<8;++j) a[r][j]-=f*a[c][j]; }
    }
    M44 R{}; for(int i=0;i<4;++i) for(int j=0;j<4;++j) R[i*4+j]=a[i][4+j]; return R;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kalman Box Tracker   state = [cx, cy, w, h, vx, vy, vw, vh]
// ─────────────────────────────────────────────────────────────────────────────

struct KalmanBoxTracker {
    V8  x;   // state
    M88 P;   // state covariance
    M88 F;   // transition (constant velocity)
    M48 H;   // observation  (we observe cx,cy,w,h)
    M88 Q;   // process noise
    M44 R;   // measurement noise

    KalmanBoxTracker() = default;

    void init(float cx, float cy, float w, float h) {
        x = {cx, cy, w, h, 0,0,0,0};

        // F: identity + velocity block
        F = eye_88();
        for(int i=0;i<4;++i) F[i*8+(i+4)] = 1.f;

        // H: observe first 4 state components
        H.fill(0);
        for(int i=0;i<4;++i) H[i*8+i] = 1.f;

        // Q: small process noise (tools are nearly static)
        Q.fill(0);
        for(int i=0;i<4;++i) Q[i*8+i]   = 1.f;     // position
        for(int i=4;i<8;++i) Q[i*8+i]   = 0.01f;   // velocity

        // R: measurement noise
        R.fill(0);
        for(int i=0;i<4;++i) R[i*4+i]   = 1.f;

        // P: initial covariance
        P.fill(0);
        for(int i=0;i<4;++i) P[i*8+i]   = 10.f;
        for(int i=4;i<8;++i) P[i*8+i]   = 100.f;   // high velocity uncertainty
    }

    void predict() {
        x          = mv_88_8(F, x);
        M88 FP     = mm_88_88(F, P);
        M88 FT     = T_88(F);
        M88 FPFT   = mm_88_88(FP, FT);
        P          = add_88(FPFT, Q);
    }

    void update(float cx, float cy, float w, float h) {
        V4  z   = {cx, cy, w, h};
        V4  Hx  = mv_48_8(H, x);
        V4  inn; for(int i=0;i<4;++i) inn[i]=z[i]-Hx[i];

        M84 HT  = T_48(H);
        M48 HP  = mm_48_88(H, P);
        M44 HPH = mm_48_84(HP, HT);
        M44 S   = add_44(HPH, R);

        M84 PHT = mm_88_84(P, HT);
        M44 Si  = inv_44(S);
        M84 K   = mm_84_44(PHT, Si);

        V8  Ky  = mv_84_4(K, inn);
        for(int i=0;i<8;++i) x[i] += Ky[i];

        M88 KH  = mm_84_48(K, H);
        M88 I   = eye_88();
        P       = mm_88_88(sub_88(I, KH), P);
    }

    // predicted box in xyxy format
    TBox predicted_box(float score, int cls) const {
        float cx=x[0], cy=x[1], w=std::max(1.f,x[2]), h=std::max(1.f,x[3]);
        return {cx-w*.5f, cy-h*.5f, cx+w*.5f, cy+h*.5f, score, cls};
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Linear assignment  (Jonker-Volgenant O(n³))
// cost[i][j] is minimised.  Returns assignment[i]=j or -1 if no valid match.
// ─────────────────────────────────────────────────────────────────────────────

inline std::vector<int> lap_solve(const std::vector<std::vector<float>>& cost,
                                   int nr, int nc) {
    if(nr==0||nc==0) return std::vector<int>(nr,-1);
    int n = std::max(nr,nc);
    const double BIG = 1e15;

    std::vector<std::vector<double>> C(n, std::vector<double>(n, BIG));
    for(int i=0;i<nr;++i) for(int j=0;j<nc;++j) C[i][j]=cost[i][j];

    std::vector<double> u(n+1,0), v(n+1,0), minv(n+1,BIG);
    std::vector<int>    p(n+1,0), way(n+1,0);
    std::vector<bool>   used(n+1,false);

    for(int i=1;i<=n;++i){
        p[0]=i; int j0=0;
        std::fill(minv.begin(),minv.end(),BIG);
        std::fill(used.begin(),used.end(),false);
        do {
            used[j0]=true;
            int i0=p[j0], j1=-1; double delta=BIG;
            for(int j=1;j<=n;++j) if(!used[j]){
                double r=C[i0-1][j-1]-u[i0]-v[j];
                if(r<minv[j]){ minv[j]=r; way[j]=j0; }
                if(minv[j]<delta){ delta=minv[j]; j1=j; }
            }
            if(j1 < 0) break;
            for(int j=0;j<=n;++j){
                if(used[j]){ u[p[j]]+=delta; v[j]-=delta; }
                else        minv[j]-=delta;
            }
            j0=j1;
        } while(p[j0]!=0);
        do{ int j1=way[j0]; p[j0]=p[j1]; j0=j1; }while(j0);
    }

    std::vector<int> res(nr,-1);
    for(int j=1;j<=n;++j)
        if(p[j]>0 && p[j]-1<nr && j-1<nc) res[p[j]-1]=j-1;
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// STrack — one tracked object instance
// ─────────────────────────────────────────────────────────────────────────────

enum class STrackState { Tracked, Lost };

struct STrack {
    inline static int next_id = 0;  // C++17 inline static

    int          id;
    STrackState  state;
    TBox         box;
    float        score;
    int          cls;
    int          time_since_update;  // frames since last detection match
    bool         is_activated;
    KalmanBoxTracker kf;

    STrack() = default;

    void init(const TBox& det) {
        id                = ++next_id;
        state             = STrackState::Tracked;
        box               = det;
        score             = det.score;
        cls               = det.cls;
        time_since_update = 0;
        is_activated      = true;
        float cx = (det.x1+det.x2)*.5f, cy = (det.y1+det.y2)*.5f;
        float w  =  det.x2-det.x1,      h  =  det.y2-det.y1;
        kf.init(cx, cy, w, h);
    }

    void predict() {
        // Zero velocity for lost tracks so they stay at last position
        if(state == STrackState::Lost)
            kf.x[4]=kf.x[5]=kf.x[6]=kf.x[7]=0.f;
        kf.predict();
        box = kf.predicted_box(score, cls);
    }

    void update(const TBox& det) {
        float cx=(det.x1+det.x2)*.5f, cy=(det.y1+det.y2)*.5f;
        float w=det.x2-det.x1,        h=det.y2-det.y1;
        kf.update(cx, cy, w, h);
        box = kf.predicted_box(det.score, det.cls);
        score             = det.score;
        cls               = det.cls;
        time_since_update = 0;
        state             = STrackState::Tracked;
    }

    TrackedBox to_tracked() const {
        return {box.x1,box.y1,box.x2,box.y2, score, cls, id};
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// BYTETracker
// ─────────────────────────────────────────────────────────────────────────────

struct BYTETracker {
    float high_thresh;    // split dets into high/low at this score
    float iou_thresh1;    // 1-IoU threshold for stage-1 matching
    float iou_thresh2;    // 1-IoU threshold for stage-2 (low-conf) matching
    float iou_thresh3;    // 1-IoU threshold for stage-3 (lost) matching
    int   max_lost;       // frames to keep a lost track before discarding

    std::vector<STrack> tracked;
    std::vector<STrack> lost;

    BYTETracker(float ht=0.5f,
                float t1=0.8f, float t2=0.5f, float t3=0.7f,
                int ml=30)
        : high_thresh(ht), iou_thresh1(t1), iou_thresh2(t2),
          iou_thresh3(t3), max_lost(ml) {}

    // Build (1-IoU) cost matrix between two sets of boxes
    static std::vector<std::vector<float>>
    iou_cost(const std::vector<STrack>& tracks,
             const std::vector<TBox>&   dets) {
        std::vector<std::vector<float>> C(tracks.size(),
                                          std::vector<float>(dets.size(), 1.f));
        for(size_t i=0;i<tracks.size();++i)
            for(size_t j=0;j<dets.size();++j)
                C[i][j] = 1.f - tbox_iou(tracks[i].box, dets[j]);
        return C;
    }

    // Same but from raw pointers (for sub-selection)
    static std::vector<std::vector<float>>
    iou_cost_ptrs(const std::vector<STrack*>& tracks,
                  const std::vector<TBox>&    dets) {
        std::vector<std::vector<float>> C(tracks.size(),
                                          std::vector<float>(dets.size(), 1.f));
        for(size_t i=0;i<tracks.size();++i)
            for(size_t j=0;j<dets.size();++j)
                C[i][j] = 1.f - tbox_iou(tracks[i]->box, dets[j]);
        return C;
    }

    std::vector<TrackedBox> update(const std::vector<TBox>& detections) {
        // ── Split detections ─────────────────────────────────────────────
        std::vector<TBox> det_hi, det_lo;
        for(const auto& d : detections)
            (d.score >= high_thresh ? det_hi : det_lo).push_back(d);

        // ── Predict all existing tracks ──────────────────────────────────
        for(auto& t : tracked) t.predict();
        for(auto& t : lost)    t.predict();

        // ── Stage 1: high-conf dets → active tracked tracks ──────────────
        std::vector<int> unmatched_trk1;
        std::vector<bool> hi_used(det_hi.size(), false);

        if(!tracked.empty() && !det_hi.empty()) {
            auto C    = iou_cost(tracked, det_hi);
            auto asgn = lap_solve(C, tracked.size(), det_hi.size());
            for(int i=0;i<(int)tracked.size();++i) {
                int j = asgn[i];
                if(j>=0 && C[i][j] <= iou_thresh1) { tracked[i].update(det_hi[j]); hi_used[j]=true; }
                else                                  unmatched_trk1.push_back(i);
            }
        } else {
            for(int i=0;i<(int)tracked.size();++i) unmatched_trk1.push_back(i);
        }

        // ── Stage 2: low-conf dets → unmatched active tracks ─────────────
        std::vector<int> still_unmatched;
        if(!unmatched_trk1.empty() && !det_lo.empty()) {
            std::vector<STrack*> ut;
            for(int i : unmatched_trk1) ut.push_back(&tracked[i]);
            auto C    = iou_cost_ptrs(ut, det_lo);
            auto asgn = lap_solve(C, ut.size(), det_lo.size());
            for(int i=0;i<(int)ut.size();++i) {
                int j = asgn[i];
                if(j>=0 && C[i][j] <= iou_thresh2) ut[i]->update(det_lo[j]);
                else                                still_unmatched.push_back(unmatched_trk1[i]);
            }
        } else {
            still_unmatched = unmatched_trk1;
        }

        // Mark still-unmatched active tracks as Lost
        for(int i : still_unmatched) { tracked[i].state = STrackState::Lost; tracked[i].time_since_update=1; }

        // Remaining high-conf detections
        std::vector<TBox> det_rem;
        for(int j=0;j<(int)det_hi.size();++j) if(!hi_used[j]) det_rem.push_back(det_hi[j]);

        // ── Stage 3: remaining high-conf dets → lost tracks ──────────────
        std::vector<bool> rem_used(det_rem.size(), false);
        if(!lost.empty() && !det_rem.empty()) {
            auto C    = iou_cost(lost, det_rem);
            auto asgn = lap_solve(C, lost.size(), det_rem.size());
            for(int i=0;i<(int)lost.size();++i) {
                int j = asgn[i];
                if(j>=0 && C[i][j] <= iou_thresh3) { lost[i].update(det_rem[j]); rem_used[j]=true; }
            }
        }

        // New tracks from fully unmatched high-conf dets
        for(int j=0;j<(int)det_rem.size();++j) if(!rem_used[j]) {
            STrack nt; nt.init(det_rem[j]); tracked.push_back(nt);
        }

        // ── Rebuild tracked / lost vectors ────────────────────────────────
        std::vector<STrack> new_tracked, new_lost;

        for(auto& t : tracked)
            (t.state == STrackState::Tracked ? new_tracked : new_lost).push_back(t);

        for(auto& t : lost) {
            if(t.state == STrackState::Tracked) {
                new_tracked.push_back(t);           // re-activated in stage 3
            } else {
                if(t.time_since_update < max_lost) {
                    ++t.time_since_update;
                    new_lost.push_back(t);
                }
                // else: silently drop (expired)
            }
        }

        tracked = std::move(new_tracked);
        lost    = std::move(new_lost);

        // ── Return active tracks ──────────────────────────────────────────
        std::vector<TrackedBox> result;
        for(const auto& t : tracked) result.push_back(t.to_tracked());
        return result;
    }
};
