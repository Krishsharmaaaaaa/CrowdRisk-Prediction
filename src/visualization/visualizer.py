"""
Visual Decision-Support Overlay and Dashboard Generator.
Renders real-time telemetry, trajectory trails, spatial density grid, and warning alerts.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import math
import numpy as np

from ..bottleneck.bottleneck_detector import BottleneckResult
from ..features.density import DensityGrid
from ..features.motion import CrowdMotionFeatures
from ..features.optical_flow import OpticalFlowFeatures
from ..panic.panic_detector import PanicResult
from ..risk.risk_fusion import RiskAssessment
from ..tracking.bytetrack_tracker import Track
from ..tracking.trajectory import TrajectoryManager
from ..warning.early_warning import EarlyWarningStatus
from ..features.cell_grid import SpatialGridFeatureMap
from ..explainability.crda_engine import GridCRDAReport
from .decision_hud import DecisionSupportHUD


class CrowdVisualizer:
    """
    Renders research-grade visual telemetry overlays for crowd safety monitoring.
    """

    COLOR_MAP = {
        "LOW": (46, 204, 113),       # Emerald Green (BGR)
        "MODERATE": (241, 196, 15),   # Yellow Amber (BGR)
        "WARNING": (230, 126, 34),    # Orange (BGR)
        "HIGH": (230, 126, 34),       # Orange (BGR)
        "CRITICAL": (231, 76, 60),    # Crimson Red (BGR)
        "ACTIVE": (231, 76, 60),
        "NORMAL": (46, 204, 113),
        "MONITORING": (241, 196, 15),
    }

    def __init__(
        self,
        draw_boxes: bool = True,
        draw_trajectories: bool = True,
        draw_density_grid: bool = True,
        draw_bottleneck_zones: bool = True,
        draw_dashboard: bool = True,
        trail_length: int = 20,
        dashboard_alpha: float = 0.75,
        enable_decision_hud: bool = True,
        decision_hud: Optional[DecisionSupportHUD] = None
    ):
        self.draw_boxes = draw_boxes
        self.draw_trajectories = draw_trajectories
        self.draw_density_grid = draw_density_grid
        self.draw_bottleneck_zones = draw_bottleneck_zones
        self.draw_dashboard = draw_dashboard
        self.trail_length = trail_length
        self.dashboard_alpha = dashboard_alpha
        self.enable_decision_hud = enable_decision_hud
        self.decision_hud = decision_hud or DecisionSupportHUD()

    def render(
        self,
        frame: np.ndarray,
        active_tracks: List[Track],
        trajectory_mgr: TrajectoryManager,
        motion_features: CrowdMotionFeatures,
        density_grid: DensityGrid,
        bottleneck_result: BottleneckResult,
        panic_result: PanicResult,
        risk_assessment: RiskAssessment,
        warning_status: EarlyWarningStatus,
        flow_features: Optional[OpticalFlowFeatures] = None,
        grid_feature_map: Optional[SpatialGridFeatureMap] = None,
        crda_report: Optional[GridCRDAReport] = None,
        selected_cell_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Renders all enabled visual layers onto a copy of the input frame.
        """
        out = frame.copy()
        h, w = canvas_shape = out.shape[:2]

        # If Decision-Support HUD is active and spatial grid or CRDA data is available,
        # render the research HUD layer
        if self.enable_decision_hud and (grid_feature_map is not None or crda_report is not None):
            # 1. Trajectory Trails Layer
            if self.draw_trajectories and trajectory_mgr is not None:
                out = self._render_trajectories(out, active_tracks, trajectory_mgr)

            # 2. Person Bounding Boxes & IDs
            if self.draw_boxes:
                out = self._render_detections(out, active_tracks, trajectory_mgr)

            # 3. Decision-Support HUD Layer (includes Spatial Heatmap, Global Risk, CRDA Drivers,
            #    Minimal Intervention Recommender, Decision Support, and Scientific Disclaimers)
            f_idx = getattr(motion_features, "frame_id", 0)
            t_sec = getattr(motion_features, "timestamp", 0.0)
            out, _ = self.decision_hud.render(
                frame=out,
                frame_idx=f_idx,
                timestamp=t_sec,
                risk_assessment=risk_assessment,
                warning_status=warning_status,
                grid_feature_map=grid_feature_map,
                crda_report=crda_report,
                selected_cell_id=selected_cell_id
            )
            return out

        # 1. Density Grid & Bottleneck Heatmap Layer
        if self.draw_density_grid and density_grid is not None:
            out = self._render_density_grid(out, density_grid)

        # 2. Local Bottleneck Zones Layer
        if self.draw_bottleneck_zones and bottleneck_result.active_zones:
            out = self._render_bottleneck_zones(out, bottleneck_result.active_zones)

        # 3. Trajectory Trails Layer
        if self.draw_trajectories and trajectory_mgr is not None:
            out = self._render_trajectories(out, active_tracks, trajectory_mgr)

        # 4. Person Bounding Boxes & IDs
        if self.draw_boxes:
            out = self._render_detections(out, active_tracks, trajectory_mgr)

        # 5. Top Warning Banner (when Active)
        if warning_status.is_warning_active:
            out = self._render_warning_banner(out, warning_status, w)

        # 6. Glassmorphic Telemetry Dashboard
        if self.draw_dashboard:
            out = self._render_dashboard(
                out,
                motion_features,
                panic_result,
                bottleneck_result,
                risk_assessment,
                warning_status,
                flow_features
            )

        return out

    def _render_density_grid(self, frame: np.ndarray, density_grid: DensityGrid) -> np.ndarray:
        overlay = frame.copy()
        for cell in density_grid.cells:
            # Draw subtle grid borders
            cv2.rectangle(overlay, (cell.x1, cell.y1), (cell.x2, cell.y2), (60, 60, 60), 1)

            # If density is elevated, add color tint
            if cell.normalized_density > 0.20:
                alpha_cell = min(0.35, float(cell.normalized_density * 0.4))
                # Interpolate between green and red
                r = int(255 * cell.normalized_density)
                g = int(255 * (1.0 - cell.normalized_density))
                color = (30, g, r)  # BGR
                cv2.rectangle(overlay, (cell.x1, cell.y1), (cell.x2, cell.y2), color, -1)

        return cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    def _render_bottleneck_zones(self, frame: np.ndarray, zones) -> np.ndarray:
        for z in zones:
            # Thick pulsing border for critical bottleneck zone
            color = (0, 140, 255) if z.score < 0.70 else (0, 0, 255)
            cv2.rectangle(frame, (z.x1, z.y1), (z.x2, z.y2), color, 2)
            label = f"BOTTLENECK {int(z.score * 100)}%"
            cv2.putText(
                frame,
                label,
                (z.x1 + 6, z.y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )
        return frame

    def _render_trajectories(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        trajectory_mgr: TrajectoryManager
    ) -> np.ndarray:
        for track in tracks:
            pts = trajectory_mgr.get_trail(track.track_id, max_points=self.trail_length)
            if len(pts) < 2:
                continue

            for i in range(1, len(pts)):
                p1 = pts[i - 1]
                p2 = pts[i]
                # Fade color with history index
                fraction = i / len(pts)
                b = int(255 * (1.0 - fraction * 0.5))
                g = int(200 * fraction)
                r = int(50 * fraction)
                thickness = 1 if i < len(pts) // 2 else 2
                cv2.line(frame, p1, p2, (b, g, r), thickness, cv2.LINE_AA)

            # Draw directional velocity pointer
            hist = list(trajectory_mgr.histories.get(track.track_id, []))
            if hist and hist[-1].speed > 0.8:
                latest = hist[-1]
                tip_x = int(round(latest.x + 4.0 * latest.dx))
                tip_y = int(round(latest.y + 4.0 * latest.dy))
                cv2.arrowedLine(
                    frame,
                    (int(round(latest.x)), int(round(latest.y))),
                    (tip_x, tip_y),
                    (0, 255, 255),
                    1,
                    tipLength=0.3
                )
        return frame

    def _render_detections(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        trajectory_mgr: TrajectoryManager
    ) -> np.ndarray:
        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.tlbr]
            # Bounding box color based on track speed
            hist = list(trajectory_mgr.histories.get(track.track_id, []))
            speed = hist[-1].speed if hist else 0.0

            if speed > 5.0:
                color = (0, 0, 255)      # Red (Sprinting)
            elif speed > 2.5:
                color = (0, 215, 255)    # Gold (Fast Walk)
            else:
                color = (0, 230, 115)    # Green (Normal Walk)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Track ID and Speed Tag
            tag = f"ID:{track.track_id} | {speed:.1f}px"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                frame,
                tag,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
        return frame

    def _render_warning_banner(
        self, frame: np.ndarray, warning_status: EarlyWarningStatus, width: int
    ) -> np.ndarray:
        banner_h = 42
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, banner_h), (0, 0, 200), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        alert_text = "CRITICAL ALERT: SUSTAINED CROWD EVACUATION / PANIC RISK DETECTED"
        (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
        tx = max(10, (width - tw) // 2)
        cv2.putText(
            frame,
            alert_text,
            (tx, (banner_h + th) // 2 - 2),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        return frame

    def _render_dashboard(
        self,
        frame: np.ndarray,
        mf: CrowdMotionFeatures,
        panic: PanicResult,
        bottleneck: BottleneckResult,
        risk: RiskAssessment,
        warning: EarlyWarningStatus,
        flow: Optional[OpticalFlowFeatures]
    ) -> np.ndarray:
        # Dashboard parameters
        pad_x, pad_y = 15, 50 if warning.is_warning_active else 15
        box_w, box_h = 320, 280

        # Semi-transparent dark background card
        sub_img = frame[pad_y: pad_y + box_h, pad_x: pad_x + box_w]
        if sub_img.shape[0] != box_h or sub_img.shape[1] != box_w:
            return frame

        card = np.full_like(sub_img, 20)  # Dark gray background
        blended = cv2.addWeighted(card, self.dashboard_alpha, sub_img, 1.0 - self.dashboard_alpha, 0)
        frame[pad_y: pad_y + box_h, pad_x: pad_x + box_w] = blended

        # Border around card
        risk_color = self.COLOR_MAP.get(risk.risk_level, (200, 200, 200))
        cv2.rectangle(frame, (pad_x, pad_y), (pad_x + box_w, pad_y + box_h), (80, 80, 80), 1)

        # Header Title
        cv2.putText(
            frame,
            "CROWD SAFETY MONITOR",
            (pad_x + 12, pad_y + 24),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        cv2.line(frame, (pad_x + 12, pad_y + 32), (pad_x + box_w - 12, pad_y + 32), (100, 100, 100), 1)

        flow_val = flow.mean_magnitude if flow else 0.0

        # Metric Lines
        lines = [
            (f"People: {mf.person_count}", (220, 220, 220)),
            (f"Mean Speed: {mf.mean_speed:.2f} px/f", (220, 220, 220)),
            (f"Density: {mf.density:.2f}", (220, 220, 220)),
            (f"Flow: {flow_val:.2f} px/f", (220, 220, 220)),
            (f"Panic Score: {panic.score:.2f} ({panic.level})", self.COLOR_MAP.get(panic.level, (220, 220, 220))),
            (f"Bottleneck Score: {bottleneck.score:.2f} ({bottleneck.level})", self.COLOR_MAP.get(bottleneck.level, (220, 220, 220))),
        ]

        curr_y = pad_y + 54
        for text, col in lines:
            cv2.putText(frame, text, (pad_x + 14, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
            curr_y += 22

        # Risk Banner inside card
        cv2.line(frame, (pad_x + 12, curr_y), (pad_x + box_w - 12, curr_y), (100, 100, 100), 1)
        curr_y += 24

        risk_str = f"RISK: {risk.overall_risk:.2f} [{risk.risk_level}]"
        cv2.putText(frame, risk_str, (pad_x + 14, curr_y), cv2.FONT_HERSHEY_DUPLEX, 0.58, risk_color, 2, cv2.LINE_AA)
        curr_y += 24

        # Early Warning Status Line
        trend_arrow = "^" if warning.risk_trend == "RISING" else ("v" if warning.risk_trend == "FALLING" else "-")
        warn_col = self.COLOR_MAP.get(warning.status_text, (200, 200, 200))
        warn_str = f"EARLY WARNING: {warning.status_text} (Trend {trend_arrow})"
        cv2.putText(frame, warn_str, (pad_x + 14, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, warn_col, 1, cv2.LINE_AA)

        return frame
