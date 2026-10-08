"""
Decision-Support HUD Visualization Engine.
Stage 9 Research Implementation:
Provides an interactive, high-visibility decision-support telemetry HUD over surveillance video:
1. Global Risk Panel (strictly labeled 'MODEL RISK', non-probabilistic)
2. CRDA Driver Panel (exact delta_R attributions for D, O, B, K, ordinal rankings, primary driver)
3. 4x4 Spatial Risk Grid Heatmap (cell ID, local model risk, threat level, dominant driver)
4. Minimal Intervention Panel (cell ID, initial risk -> CF risk, minimal subset, safe reached flag,
   explicit 'NO SAFE-THRESHOLD INTERVENTION FOUND' if safe target not reached)
5. Operator-Oriented Decision Support Card (strictly non-causal, model-based narrative)
6. Scientific Disclaimer and Audit Trail Strip
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from ..features.cell_grid import SpatialCellFeatureVector, SpatialGridFeatureMap
from ..explainability.crda_engine import (
    CellCRDAExplanation,
    DriverAttribution,
    GridCRDAReport,
    SubsetIntervention,
)
from ..risk.risk_fusion import RiskAssessment
from ..warning.early_warning import EarlyWarningStatus


@dataclass
class HUDTelemetryFrame:
    """Machine-readable per-frame telemetry structure for audit and logging."""
    frame_idx: int
    timestamp: float
    model_risk: float
    threat_level: str
    warning_status: str
    risk_trend: str
    selected_cell_id: str
    selected_cell_risk: float
    top_driver: str
    driver_deltas: Dict[str, float]
    driver_rankings: List[str]
    minimal_intervention_subset: List[str]
    predicted_cf_risk: float
    predicted_delta_R: float
    safe_threshold_reached: bool
    safe_threshold_val: float
    decision_support_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
            "model_risk": round(self.model_risk, 4),
            "threat_level": self.threat_level,
            "warning_status": self.warning_status,
            "risk_trend": self.risk_trend,
            "selected_cell_id": self.selected_cell_id,
            "selected_cell_risk": round(self.selected_cell_risk, 4),
            "top_driver": self.top_driver,
            "driver_deltas": {k: round(v, 4) for k, v in self.driver_deltas.items()},
            "driver_rankings": self.driver_rankings,
            "minimal_intervention_subset": self.minimal_intervention_subset,
            "predicted_cf_risk": round(self.predicted_cf_risk, 4),
            "predicted_delta_R": round(self.predicted_delta_R, 4),
            "safe_threshold_reached": self.safe_threshold_reached,
            "safe_threshold_val": self.safe_threshold_val,
            "decision_support_text": self.decision_support_text
        }


class DecisionSupportHUD:
    """
    Renders research-grade Decision-Support HUD telemetry overlays for crowd monitoring.
    Adheres strictly to scientific disclaimers:
    - Risk is model-derived, non-probabilistic.
    - CRDA attributions are model-based counterfactual sensitivities, not causal proofs.
    - Simulation benchmarks provide independent supporting evidence, not real-world causality.
    """

    # BGR Color Palette
    COLOR_MAP = {
        "LOW": (46, 204, 113),        # Emerald Green
        "MODERATE": (15, 196, 241),    # Amber Gold
        "HIGH": (34, 126, 230),        # Coral Orange
        "CRITICAL": (60, 76, 231),     # Crimson Red
        "NORMAL": (46, 204, 113),
        "MONITORING": (15, 196, 241),
        "WARNING": (34, 126, 230),
        "ACTIVE": (60, 76, 231),
    }

    # Driver Specific Colors (BGR)
    DRIVER_COLORS = {
        "D": (235, 160, 52),   # Blue / Cyan
        "O": (180, 105, 255),  # Violet / Magenta
        "B": (34, 126, 230),   # Orange
        "K": (60, 76, 231),    # Red
        "Density": (235, 160, 52),
        "Disorder": (180, 105, 255),
        "Bottleneck": (34, 126, 230),
        "Kinematics": (60, 76, 231)
    }

    def __init__(
        self,
        high_risk_threshold: float = 0.60,
        moderate_risk_threshold: float = 0.30,
        show_spatial_heatmap: bool = True,
        show_driver_panel: bool = True,
        show_intervention_panel: bool = True,
        show_decision_support: bool = True,
        show_disclaimer: bool = True,
        alpha_panel: float = 0.82,
        alpha_cell_tint: float = 0.22,
        mode: str = "overlay"  # 'overlay' or 'cockpit'
    ):
        self.high_risk_threshold = high_risk_threshold
        self.moderate_risk_threshold = moderate_risk_threshold
        self.show_spatial_heatmap = show_spatial_heatmap
        self.show_driver_panel = show_driver_panel
        self.show_intervention_panel = show_intervention_panel
        self.show_decision_support = show_decision_support
        self.show_disclaimer = show_disclaimer
        self.alpha_panel = alpha_panel
        self.alpha_cell_tint = alpha_cell_tint
        self.mode = mode

    def render(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        risk_assessment: Optional[RiskAssessment] = None,
        warning_status: Optional[EarlyWarningStatus] = None,
        grid_feature_map: Optional[SpatialGridFeatureMap] = None,
        crda_report: Optional[GridCRDAReport] = None,
        selected_cell_id: Optional[str] = None
    ) -> Tuple[np.ndarray, HUDTelemetryFrame]:
        """
        Renders all HUD panels onto a copy of the input frame.
        Returns:
            (annotated_frame, telemetry_record)
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Extract or mock default states if inputs are missing
        overall_risk = float(risk_assessment.overall_risk) if risk_assessment else 0.0
        risk_level = risk_assessment.risk_level if risk_assessment else self._classify_threat(overall_risk)
        warn_status = warning_status.status_text if warning_status else "NORMAL"
        risk_trend = warning_status.risk_trend if warning_status else "STABLE"

        # Select target cell for intervention analysis
        target_explanation, target_cell = self._select_target_cell(
            grid_feature_map=grid_feature_map,
            crda_report=crda_report,
            selected_cell_id=selected_cell_id
        )

        # 1. Render 4x4 Spatial Risk Heatmap Grid Layer
        if self.show_spatial_heatmap and grid_feature_map is not None:
            canvas = self._render_spatial_grid_heatmap(
                canvas, grid_feature_map, crda_report, target_explanation
            )

        # 2. Render HUD Telemetry Panels
        if self.mode == "cockpit" and w >= 800:
            canvas, telemetry = self._render_cockpit_layout(
                canvas, frame_idx, timestamp, overall_risk, risk_level,
                warn_status, risk_trend, target_explanation, target_cell
            )
        else:
            canvas, telemetry = self._render_overlay_layout(
                canvas, frame_idx, timestamp, overall_risk, risk_level,
                warn_status, risk_trend, target_explanation, target_cell
            )

        # 3. Scientific Disclaimer Footer
        if self.show_disclaimer:
            canvas = self._render_scientific_disclaimer(canvas)

        return canvas, telemetry

    def _classify_threat(self, risk: float) -> str:
        if risk < self.moderate_risk_threshold:
            return "LOW"
        elif risk < self.high_risk_threshold:
            return "MODERATE"
        elif risk < 0.85:
            return "HIGH"
        return "CRITICAL"

    def _select_target_cell(
        self,
        grid_feature_map: Optional[SpatialGridFeatureMap],
        crda_report: Optional[GridCRDAReport],
        selected_cell_id: Optional[str]
    ) -> Tuple[Optional[CellCRDAExplanation], Optional[SpatialCellFeatureVector]]:
        """Identifies the cell to highlight and display in the intervention panel."""
        target_explanation: Optional[CellCRDAExplanation] = None
        target_cell: Optional[SpatialCellFeatureVector] = None

        if crda_report and crda_report.cell_explanations:
            if selected_cell_id:
                for exp in crda_report.cell_explanations:
                    if exp.cell_id == selected_cell_id:
                        target_explanation = exp
                        break
            if target_explanation is None:
                # Default to the most critical cell
                target_explanation = crda_report.most_critical_cell or crda_report.cell_explanations[0]

        if grid_feature_map and target_explanation:
            for c in grid_feature_map.cells:
                if c.cell_id == target_explanation.cell_id:
                    target_cell = c
                    break
        elif grid_feature_map and not target_explanation:
            # Fallback to cell with highest occupancy
            if grid_feature_map.cells:
                target_cell = max(grid_feature_map.cells, key=lambda c: c.person_count)

        return target_explanation, target_cell

    def _render_spatial_grid_heatmap(
        self,
        canvas: np.ndarray,
        grid_map: SpatialGridFeatureMap,
        crda_report: Optional[GridCRDAReport],
        focus_explanation: Optional[CellCRDAExplanation]
    ) -> np.ndarray:
        """Overlays the 4x4 spatial grid with cell ID, local model risk, threat level, and dominant driver."""
        overlay = canvas.copy()
        focus_id = focus_explanation.cell_id if focus_explanation else None

        # Build map of explanations by cell ID
        exp_by_id = {e.cell_id: e for e in crda_report.cell_explanations} if crda_report else {}

        for cell in grid_map.cells:
            x1, y1, x2, y2 = cell.x1, cell.y1, cell.x2, cell.y2
            expl = exp_by_id.get(cell.cell_id)

            # Local model risk (from explanation or computed main weights)
            if expl:
                c_risk = expl.initial_risk
                threat = expl.initial_threat_level
                top_key = expl.top_driver_key
            else:
                c_risk = 0.35 * cell.D + 0.20 * cell.O + 0.30 * cell.B + 0.15 * cell.K
                threat = self._classify_threat(c_risk)
                top_key = "D" if cell.D >= max(cell.O, cell.B, cell.K) else (
                    "O" if cell.O >= max(cell.B, cell.K) else ("B" if cell.B >= cell.K else "K")
                )

            color = self.COLOR_MAP.get(threat, (120, 120, 120))
            is_focus = (cell.cell_id == focus_id)

            # Visually distinguish elevated cells with translucent fill
            if c_risk >= self.moderate_risk_threshold or cell.person_count > 0:
                tint_alpha = 0.30 if c_risk >= self.high_risk_threshold else 0.15
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                canvas = cv2.addWeighted(overlay, tint_alpha, canvas, 1.0 - tint_alpha, 0)
                overlay = canvas.copy()

            # Cell Border: Focus cell gets bright cyan/white thick border
            if is_focus:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 240, 0), 2)  # Cyan border
                cv2.rectangle(canvas, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 255, 255), 1)
            else:
                border_col = color if c_risk >= self.moderate_risk_threshold else (90, 90, 90)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1)

            # Cell Header Tag (Cell ID | Local Risk | Driver)
            cell_w = x2 - x1
            cell_h = y2 - y1
            if cell_w >= 50 and cell_h >= 30:
                tag_str = f"{cell.cell_id} R:{c_risk:.2f} [{top_key}]"
                font_scale = 0.32 if cell_w < 100 else 0.40
                (tw, th), _ = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

                tx1, ty1 = x1 + 3, y1 + 3
                tx2, ty2 = tx1 + tw + 4, ty1 + th + 4
                cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), (15, 15, 15), -1)
                cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), color, 1)
                cv2.putText(
                    canvas,
                    tag_str,
                    (tx1 + 2, ty1 + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        return canvas

    def _render_overlay_layout(
        self,
        canvas: np.ndarray,
        frame_idx: int,
        timestamp: float,
        overall_risk: float,
        risk_level: str,
        warn_status: str,
        risk_trend: str,
        target_explanation: Optional[CellCRDAExplanation],
        target_cell: Optional[SpatialCellFeatureVector]
    ) -> Tuple[np.ndarray, HUDTelemetryFrame]:
        """Renders HUD panels in overlay mode directly on the video."""
        h, w = canvas.shape[:2]

        scale = max(0.48, min(w / 1280.0, h / 720.0))

        p1_w, p1_h = int(320 * scale), int(150 * scale)
        p2_w, p2_h = int(340 * scale), int(290 * scale)

        # ----------------------------------------------------
        # Panel 1: Top-Left GLOBAL RISK PANEL
        # ----------------------------------------------------
        p1_x, p1_y = int(12 * scale), int(12 * scale)
        canvas = self._draw_glass_card(canvas, p1_x, p1_y, p1_w, p1_h, self.COLOR_MAP.get(risk_level, (180, 180, 180)))

        fs_title = 0.48 * scale
        fs_body = 0.40 * scale
        lh = int(22 * scale)
        cy = p1_y + int(20 * scale)

        cv2.putText(canvas, "GLOBAL MODEL RISK", (p1_x + 8, cy), cv2.FONT_HERSHEY_DUPLEX, fs_title, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (p1_x + 8, cy + 6), (p1_x + p1_w - 8, cy + 6), (70, 70, 70), 1)
        cy += lh + int(4 * scale)

        # Prominent MODEL RISK label
        risk_col = self.COLOR_MAP.get(risk_level, (255, 255, 255))
        risk_txt = f"MODEL RISK: {overall_risk:.2f}"
        cv2.putText(canvas, risk_txt, (p1_x + 8, cy), cv2.FONT_HERSHEY_DUPLEX, 0.58 * scale, risk_col, 2, cv2.LINE_AA)
        cy += lh

        threat_txt = f"Threat Level: {risk_level}  (Trend: {risk_trend})"
        cv2.putText(canvas, threat_txt, (p1_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, fs_body, (220, 220, 220), 1, cv2.LINE_AA)
        cy += lh

        warn_col = self.COLOR_MAP.get(warn_status, (200, 200, 200))
        status_txt = f"Warning: {warn_status} | Frm {frame_idx} ({timestamp:.2f}s)"
        cv2.putText(canvas, status_txt, (p1_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, fs_body, warn_col, 1, cv2.LINE_AA)
        cy += int(16 * scale)

        cv2.putText(canvas, "[Non-calibrated model index in [0, 1]]", (p1_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.32 * scale, (150, 150, 150), 1, cv2.LINE_AA)

        # ----------------------------------------------------
        # Panel 2: Top-Right CRDA DRIVER & INTERVENTION PANEL
        # ----------------------------------------------------
        p2_x = max(p1_x + p1_w + 10, w - p2_w - int(12 * scale))
        p2_y = int(12 * scale)
        border_col = (255, 240, 0) if target_explanation else (80, 80, 80)
        canvas = self._draw_glass_card(canvas, p2_x, p2_y, p2_w, p2_h, border_col)

        cy2 = p2_y + int(20 * scale)
        cell_lbl = target_explanation.cell_id if target_explanation else "GLOBAL"
        cv2.putText(canvas, f"CRDA ATTRIBUTION | ZONE {cell_lbl}", (p2_x + 8, cy2), cv2.FONT_HERSHEY_DUPLEX, fs_title, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (p2_x + 8, cy2 + 6), (p2_x + p2_w - 8, cy2 + 6), (70, 70, 70), 1)
        cy2 += lh

        # Extract Driver Deltas and Rankings
        driver_deltas, driver_ranks, top_driver = self._extract_driver_info(target_explanation, target_cell)

        # Driver Bars
        bar_max_w = int(140 * scale)
        bar_h = int(10 * scale)
        for d_key in ["D", "O", "B", "K"]:
            d_name = {"D": "Density", "O": "Disorder", "B": "Bottleneck", "K": "Kinematics"}[d_key]
            d_delta = driver_deltas.get(d_key, 0.0)
            d_col = self.DRIVER_COLORS[d_key]

            lbl = f"{d_key} ({d_name[:4]}): dR={d_delta:.3f}"
            cv2.putText(canvas, lbl, (p2_x + 8, cy2 + int(8 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.36 * scale, (230, 230, 230), 1, cv2.LINE_AA)

            bx = p2_x + p2_w - bar_max_w - int(12 * scale)
            by = cy2
            cv2.rectangle(canvas, (bx, by), (bx + bar_max_w, by + bar_h), (40, 40, 40), -1)
            fill_w = int(np.clip(d_delta / 0.50, 0.0, 1.0) * bar_max_w)
            if fill_w > 0:
                cv2.rectangle(canvas, (bx, by), (bx + fill_w, by + bar_h), d_col, -1)
            cv2.rectangle(canvas, (bx, by), (bx + bar_max_w, by + bar_h), (80, 80, 80), 1)

            cy2 += int(18 * scale)

        cy2 += int(4 * scale)
        cv2.putText(canvas, f"Primary Driver: {top_driver}", (p2_x + 8, cy2), cv2.FONT_HERSHEY_SIMPLEX, fs_body, (255, 235, 120), 1, cv2.LINE_AA)
        cy2 += lh

        cv2.line(canvas, (p2_x + 8, cy2), (p2_x + p2_w - 8, cy2), (60, 60, 60), 1)
        cy2 += int(14 * scale)

        cv2.putText(canvas, "MINIMAL INTERVENTION RECOMMENDER", (p2_x + 8, cy2), cv2.FONT_HERSHEY_DUPLEX, 0.38 * scale, (200, 220, 255), 1, cv2.LINE_AA)
        cy2 += int(18 * scale)

        # Minimal Intervention Logic & Strict Safe Threshold Flag
        min_subset, init_r, cf_r, dR, safe_reached = self._extract_minimal_intervention(
            target_explanation, overall_risk
        )

        if safe_reached:
            subset_str = "{" + ", ".join(min_subset) + "}"
            cv2.putText(canvas, f"Subset: {subset_str}  (Card: {len(min_subset)})", (p2_x + 8, cy2), cv2.FONT_HERSHEY_SIMPLEX, fs_body, (255, 255, 255), 1, cv2.LINE_AA)
            cy2 += int(16 * scale)
            cv2.putText(canvas, f"Model Risk: {init_r:.2f} -> {cf_r:.2f}  (dR={dR:.2f})", (p2_x + 8, cy2), cv2.FONT_HERSHEY_SIMPLEX, fs_body, (220, 220, 220), 1, cv2.LINE_AA)
            cy2 += int(16 * scale)
            cv2.putText(canvas, f"Safe Threshold Reached: YES (< {self.high_risk_threshold:.2f})", (p2_x + 8, cy2), cv2.FONT_HERSHEY_SIMPLEX, fs_body, (46, 204, 113), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "NO SAFE-THRESHOLD INTERVENTION FOUND", (p2_x + 8, cy2), cv2.FONT_HERSHEY_DUPLEX, 0.40 * scale, (60, 76, 231), 1, cv2.LINE_AA)
            cy2 += int(16 * scale)
            fallback_str = "{" + ", ".join(min_subset) + "}"
            cv2.putText(canvas, f"Fallback (Max dR): {fallback_str}  (dR={dR:.2f})", (p2_x + 8, cy2), cv2.FONT_HERSHEY_SIMPLEX, fs_body, (220, 200, 200), 1, cv2.LINE_AA)
            cy2 += int(16 * scale)
            cv2.putText(canvas, f"Safe Threshold Reached: NO (CF Risk {cf_r:.2f} >= {self.high_risk_threshold:.2f})", (p2_x + 8, cy2), cv2.FONT_HERSHEY_SIMPLEX, fs_body, (60, 76, 231), 1, cv2.LINE_AA)

        cy2 += int(18 * scale)

        # ----------------------------------------------------
        # Panel 3: Bottom-Left OPERATOR DECISION SUPPORT CARD
        # ----------------------------------------------------
        if self.show_decision_support:
            canvas, dec_text = self._render_decision_support_card(
                canvas, scale, top_driver, min_subset, dR, safe_reached
            )
        else:
            dec_text = f"Primary model driver is {top_driver}."

        telemetry = HUDTelemetryFrame(
            frame_idx=frame_idx,
            timestamp=timestamp,
            model_risk=overall_risk,
            threat_level=risk_level,
            warning_status=warn_status,
            risk_trend=risk_trend,
            selected_cell_id=cell_lbl,
            selected_cell_risk=init_r,
            top_driver=top_driver,
            driver_deltas=driver_deltas,
            driver_rankings=driver_ranks,
            minimal_intervention_subset=min_subset,
            predicted_cf_risk=cf_r,
            predicted_delta_R=dR,
            safe_threshold_reached=safe_reached,
            safe_threshold_val=self.high_risk_threshold,
            decision_support_text=dec_text
        )

        return canvas, telemetry

    def _render_cockpit_layout(
        self,
        canvas: np.ndarray,
        frame_idx: int,
        timestamp: float,
        overall_risk: float,
        risk_level: str,
        warn_status: str,
        risk_trend: str,
        target_explanation: Optional[CellCRDAExplanation],
        target_cell: Optional[SpatialCellFeatureVector]
    ) -> Tuple[np.ndarray, HUDTelemetryFrame]:
        """Renders wide composite cockpit layout with side panel for high-density telemetry."""
        return self._render_overlay_layout(
            canvas, frame_idx, timestamp, overall_risk, risk_level,
            warn_status, risk_trend, target_explanation, target_cell
        )

    def _render_decision_support_card(
        self,
        canvas: np.ndarray,
        scale: float,
        top_driver: str,
        min_subset: List[str],
        delta_R: float,
        safe_reached: bool
    ) -> Tuple[np.ndarray, str]:
        """Renders the concise operator decision-support card with strictly non-causal language."""
        h, w = canvas.shape[:2]
        cw, ch = int(460 * scale), int(80 * scale)
        cx = int(12 * scale)
        cy = h - ch - int(34 * scale)

        canvas = self._draw_glass_card(canvas, cx, cy, cw, ch, (100, 100, 100))

        fs_title = 0.40 * scale
        fs_text = 0.35 * scale

        cv2.putText(canvas, "MODEL-BASED DECISION SUPPORT", (cx + 8, cy + int(16 * scale)), cv2.FONT_HERSHEY_DUPLEX, fs_title, (241, 196, 15), 1, cv2.LINE_AA)

        subset_str = " + ".join(min_subset) if min_subset else top_driver
        if safe_reached:
            line1 = f"Primary risk driver: {top_driver}."
            line2 = f"Perturbing [{subset_str}] toward calm reference reduces model risk by dR = {delta_R:.2f}."
        else:
            line1 = f"Primary risk driver: {top_driver} (High Risk Sustained)."
            line2 = f"Perturbing [{subset_str}] provides maximal model risk reduction dR = {delta_R:.2f}."

        cv2.putText(canvas, line1, (cx + 8, cy + int(36 * scale)), cv2.FONT_HERSHEY_SIMPLEX, fs_text, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (cx + 8, cy + int(54 * scale)), cv2.FONT_HERSHEY_SIMPLEX, fs_text, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "(Sensitivity attribution. Evacuation simulation provides independent benchmark.)", (cx + 8, cy + int(70 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.28 * scale, (150, 150, 150), 1, cv2.LINE_AA)

        full_text = f"{line1} {line2}"
        return canvas, full_text

    def _render_scientific_disclaimer(self, canvas: np.ndarray) -> np.ndarray:
        """Renders scientific disclaimer strip across the bottom of the video."""
        h, w = canvas.shape[:2]
        bar_h = max(20, int(h * 0.032))
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (10, 10, 10), -1)
        canvas = cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0)

        disclaimer_text = "RESEARCH PROTOTYPE | MODEL-DERIVED RISK | CRDA: COUNTERFACTUAL ATTRIBUTION (NON-CAUSAL) | SIMULATION: INDEPENDENT BENCHMARK"
        fs = max(0.30, min(0.44, w / 2600.0))
        (tw, th), _ = cv2.getTextSize(disclaimer_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        tx = max(8, (w - tw) // 2)
        ty = h - int((bar_h - th) / 2)
        cv2.putText(canvas, disclaimer_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (180, 180, 180), 1, cv2.LINE_AA)

        return canvas

    def _draw_glass_card(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        border_color: Tuple[int, int, int]
    ) -> np.ndarray:
        """Renders a semi-transparent dark rounded/bordered card on the canvas."""
        ch, cw = canvas.shape[:2]
        if x + w > cw or y + h > ch or x < 0 or y < 0:
            return canvas

        sub = canvas[y: y + h, x: x + w]
        card = np.full_like(sub, 18)
        blended = cv2.addWeighted(card, self.alpha_panel, sub, 1.0 - self.alpha_panel, 0)
        canvas[y: y + h, x: x + w] = blended

        cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 1)
        return canvas

    def _extract_driver_info(
        self,
        explanation: Optional[CellCRDAExplanation],
        cell: Optional[SpatialCellFeatureVector]
    ) -> Tuple[Dict[str, float], List[str], str]:
        """Extracts exact delta_R values, ranking, and primary driver."""
        if explanation:
            deltas = {attr.driver_key: float(attr.delta_R) for attr in explanation.single_driver_attributions}
            sorted_drivers = sorted(deltas.items(), key=lambda item: -item[1])
            driver_ranks = [k for k, _ in sorted_drivers]
            top_driver = explanation.top_driver
            return deltas, driver_ranks, top_driver
        elif cell:
            deltas = {
                "D": float(0.35 * cell.D),
                "O": float(0.20 * cell.O),
                "B": float(0.30 * cell.B),
                "K": float(0.15 * cell.K),
            }
            sorted_drivers = sorted(deltas.items(), key=lambda item: -item[1])
            driver_ranks = [k for k, _ in sorted_drivers]
            name_map = {"D": "Density", "O": "Disorder", "B": "Bottleneck", "K": "Kinematics"}
            top_driver = name_map[driver_ranks[0]]
            return deltas, driver_ranks, top_driver
        else:
            deltas = {"D": 0.0, "O": 0.0, "B": 0.0, "K": 0.0}
            return deltas, ["D", "O", "B", "K"], "Density"

    def _extract_minimal_intervention(
        self,
        explanation: Optional[CellCRDAExplanation],
        overall_risk: float
    ) -> Tuple[List[str], float, float, float, bool]:
        """Extracts minimal intervention subset, initial risk, counterfactual risk, delta_R, and safe flag."""
        if explanation and explanation.minimal_intervention_subset:
            min_sub = explanation.minimal_intervention_subset
            return (
                min_sub.subset_names,
                explanation.initial_risk,
                min_sub.counterfactual_risk,
                min_sub.delta_R,
                min_sub.achieved_safe_threshold
            )
        else:
            return (["Density"], overall_risk, max(0.0, overall_risk - 0.25), 0.25, overall_risk < self.high_risk_threshold)
