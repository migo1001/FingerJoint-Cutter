"""
Finger Joint Generator for FreeCAD — Clean, Documented, Single-File

This macro creates finger joints between intersecting parts and generates an optimized
2D projection for laser cutting. It features a GUI to pick parts, set order, and tune kerf.

Key properties:
- Only **visible** solids are considered.
- Non-overlapping intersections: once a pair (A,B) is processed, its intersection volume
  is marked as "used" so later pairs cannot reuse it.
- Robust, axis-aligned finger generation along the intersection’s primary axis only.
- Kerf applied **symmetrically** along internal boundaries (kerf/2 to neighbors), preserving spacing.
- Compact, readable structure (PEP-8), full docstrings, and persistent file logging mirrored to the console.

Table of contents:
  1) Imports and Qt compatibility
  2) Constants and small data classes
  3) Logging + vector helpers
  4) GUI: FingerJointDialog (ordering + kerf)
  5) Geometry: Intersection + finger creation
  6) 2D projection/packing: ProjectionOrchestrator
  7) Progress helper
  8) Orchestrator: FingerJointOrchestrator (refactored pipeline)
  9) Main entry
"""

# =============================================================================
# 1) Imports and Qt compatibility
# =============================================================================

import FreeCAD
import Part
import FreeCADGui
import math
import traceback
import re
from typing import List, Optional, Tuple, Dict, Set, Any
from enum import Enum
from dataclasses import dataclass

# Qt (Qt6 → PySide6, fallback to PySide2)
try:
    from PySide6 import QtWidgets, QtCore
    FreeCAD.Console.PrintMessage("Successfully imported PySide6 (Qt6).\n")
except ImportError:
    try:
        from PySide2 import QtWidgets, QtCore
        FreeCAD.Console.PrintMessage("Successfully imported PySide2 (Qt5).\n")
    except ImportError:
        msg = (
            "Could not import QtWidgets from either PySide6 or PySide2. "
            "Your FreeCAD installation's Python environment may be misconfigured or incomplete."
        )
        FreeCAD.Console.PrintError(msg + "\n")
        raise ImportError(msg)


# =============================================================================
# 2) Constants and small data classes
# =============================================================================

# Tunables / defaults
DEFAULT_MIN_FINGER_LENGTH = 30.0   # mm
DEFAULT_KERF_MM = 0.135            # mm
GEOM_EPS = 1e-6
PROJECTION_MARGIN = 1.0            # mm between packed items
LAYOUT_GAP = 50.0                  # mm between thickness groups
MIN_FINGER_COUNT = 3
CROSS_JOINT_FINGER_COUNT = 2
PREFS_GROUP = "User parameter:BaseApp/Preferences/Macros/FingerJointCutter"
PREF_KERF = "KerfMM"
PREF_FINGER = "MinFingerLength"


@dataclass
class GlobalSettings:
    """Global settings shared by every intersection."""
    kerf: float
    finger_length: float


class FingerJointError(Exception):
    """Base exception for all finger joint processing errors."""


class IntersectionError(FingerJointError):
    """Raised when geometric intersections cannot be computed."""


class ProjectionError(FingerJointError):
    """Raised when 2D projection generation fails."""


class DocumentUpdateError(FingerJointError):
    """Raised when FreeCAD document updates fail."""


class ProgressError(FingerJointError):
    """Raised when progress dialog operations fail."""


class UserCancelledError(FingerJointError):
    """Raised when the user cancels processing."""

def log(msg: str,
        level: Any = 'info',
        end: str = '\n',
        exc_cls: Optional[type[FingerJointError]] = None,
        original: Optional[BaseException] = None) -> None:
    """Console logging wrapper that never raises exceptions."""
    if isinstance(level, (int, float)):
        level = 'error' if level <= 0 else 'info'
    normalized = str(level).lower()
    if normalized == 'warning':
        FreeCAD.Console.PrintWarning(f"{msg}{end}")
        return
    if normalized == 'error':
        FreeCAD.Console.PrintError(f"{msg}{end}")
        return
    FreeCAD.Console.PrintMessage(f"{msg}{end}")


def fail(msg: str,
         exc_cls: type[FingerJointError] = FingerJointError,
         original: Optional[BaseException] = None) -> None:
    """Surface a blocking error via modal dialog and raise the provided exception."""
    FreeCAD.Console.PrintError(f"{msg}\n")
    try:
        from PySide6 import QtWidgets as _QtWidgets
    except ImportError:
        try:
            from PySide2 import QtWidgets as _QtWidgets
        except ImportError:
            if original:
                raise exc_cls(msg) from original
            raise exc_cls(msg)
    _QtWidgets.QMessageBox.critical(None, "Finger Cutter", msg)
    if original:
        raise exc_cls(msg) from original
    raise exc_cls(msg)



# =============================================================================
# 3) Logging + vector helpers
# =============================================================================

log("Finger joint generator module loaded.")


# =============================================================================
# 4) GUI: FingerJointDialog (global kerf + finger length)
# =============================================================================

class FingerJointDialog(QtWidgets.QDialog):
    """Minimal dialog to capture global kerf and finger length."""

    def __init__(self,
                 doc: FreeCAD.Document,
                 parent=None):
        """Construct the streamlined parameter dialog."""
        super().__init__(parent)
        self.doc = doc
        self.setWindowTitle("Finger Joint — Parameters")
        self.setMinimumWidth(360)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(QtWidgets.QLabel(
            "Enter the global kerf and the minimum finger length.\n"
            "All visible solids inside the selected container are processed"
            " exactly in their FreeCAD tree order."
        ))

        form = QtWidgets.QFormLayout()

        saved_kerf, saved_finger = self._load_saved_settings()

        self.kerf_spin = QtWidgets.QDoubleSpinBox()
        self.kerf_spin.setRange(0.0, 5.0)
        self.kerf_spin.setDecimals(3)
        self.kerf_spin.setValue(saved_kerf)
        self.kerf_spin.setSuffix(" mm")
        form.addRow("Kerf (laser width)", self.kerf_spin)

        self.finger_spin = QtWidgets.QDoubleSpinBox()
        self.finger_spin.setRange(1.0, 1000.0)
        self.finger_spin.setDecimals(2)
        self.finger_spin.setValue(saved_finger)
        self.finger_spin.setSuffix(" mm")
        form.addRow("Minimum finger length", self.finger_spin)

        main_layout.addLayout(form)
        main_layout.addStretch()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

    def get_values(self) -> Optional[GlobalSettings]:
        """Return the chosen global settings or None on cancellation."""
        if self.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        kerf = self.kerf_spin.value()
        finger_length = self.finger_spin.value()
        self._persist_settings(kerf, finger_length)
        return GlobalSettings(kerf=kerf, finger_length=finger_length)

    @staticmethod
    def _load_saved_settings() -> Tuple[float, float]:
        """Return persisted kerf and finger length, falling back to defaults."""
        kerf = DEFAULT_KERF_MM
        finger = DEFAULT_MIN_FINGER_LENGTH
        try:
            params = FreeCAD.ParamGet(PREFS_GROUP)
            kerf = float(params.GetFloat(PREF_KERF, kerf))
            finger = float(params.GetFloat(PREF_FINGER, finger))
        except Exception:
            log("Preferences unavailable; using defaults for kerf and finger length.", 'warning')
            return kerf, finger

        kerf = kerf if kerf >= 0.0 else DEFAULT_KERF_MM
        finger = finger if finger >= 1.0 else DEFAULT_MIN_FINGER_LENGTH
        return kerf, finger

    @staticmethod
    def _persist_settings(kerf: float, finger_length: float) -> None:
        """Store kerf and finger length for reuse in future runs."""
        try:
            params = FreeCAD.ParamGet(PREFS_GROUP)
            params.SetFloat(PREF_KERF, float(kerf))
            params.SetFloat(PREF_FINGER, float(finger_length))
        except Exception:
            log("Failed to persist settings; defaults will be used next time.", 'warning')


# =============================================================================
# 5) Geometry: Intersection + finger creation
# =============================================================================

class IntersectionType(Enum):
    """Heuristic type for an intersection between two solids."""
    CORNER = "corner"
    FACE = "face"
    CROSS = "cross"


class Intersection:
    """Analyze a pair of parts and manage the finger-cut workflow."""

    _AXIS_VECTORS = (
        FreeCAD.Vector(1.0, 0.0, 0.0),
        FreeCAD.Vector(0.0, 1.0, 0.0),
        FreeCAD.Vector(0.0, 0.0, 1.0),
    )

    def __init__(self,
                 part1: FreeCAD.DocumentObject,
                 part2: FreeCAD.DocumentObject,
                 globals_: GlobalSettings,
                 used_space: Optional[Part.Shape]):
        self.part1 = part1
        self.part2 = part2
        self.globals = globals_
        self.finger_length = max(1.0, float(globals_.finger_length))
        self.kerf = max(0.0, float(globals_.kerf))
        self.used_space = used_space

        self.shape: Optional[Part.Shape] = None
        self.volume: float = 0.0
        self.bbox: Optional[Part.BoundBox] = None
        self.primary_axis_idx: int = 0
        self.longest_edge_len: float = 0.0
        self.intersection_type: IntersectionType = IntersectionType.CROSS

        self._has_geometry = False
        self._finger_boxes: List[Part.Shape] = []


    def process(self,
                ) -> Optional[Part.Shape]:
        """
        Compute intersection, generate fingers, and cut parts in one step.
        Returns updated used_space.
        """
        self.update_used_space()
        if not self.is_valid():
            return self.used_space
        self.generate_fingers()
        if not self.is_valid_fingers():
            return self.used_space
        new1, new2 = self.cut_fingers(self.part1.Shape, self.part2.Shape)
        self.part1.Shape = new1
        self.part2.Shape = new2
        return self.used_space



    # --- Calculation ----------------------------------------------------------

    def update_used_space(self) -> Optional[Part.Shape]:
        """Compute the trimmed intersection and update the used-space mask."""
        raw = self._pairwise_common()
        if raw is None or not raw.isValid() or raw.Volume < GEOM_EPS:
            self._has_geometry = False
            return self.used_space

        trimmed, updated_space = self._trim_against_used_space(raw, self.used_space)
        if trimmed is None:
            self._has_geometry = False
            self.used_space = updated_space
            return self.used_space

        self._populate_from_shape(trimmed)
        self._has_geometry = True
        self.used_space = updated_space
        return self.used_space

    def is_valid(self) -> bool:
        return self._has_geometry

    def _populate_from_shape(self, shape: Part.Shape) -> None:
        self.shape = shape
        self.volume = float(shape.Volume)
        self.bbox = shape.BoundBox
        dims = [self.bbox.XLength, self.bbox.YLength, self.bbox.ZLength]
        self.primary_axis_idx = self._dominant_axis_index(dims)
        self.longest_edge_len = float(dims[self.primary_axis_idx])
        self._determine_type()

    def _pairwise_common(self) -> Optional[Part.Shape]:
        try:
            return self.part1.Shape.common(self.part2.Shape)
        except Part.OCCError as exc:
            fail(
                f"Failed to compute pairwise intersection between '{self.part1.Label}' and '{self.part2.Label}'",
                IntersectionError,
                exc
            )

    def _trim_against_used_space(self,
                                 shape: Part.Shape,
                                 used_space: Optional[Part.Shape]) -> Tuple[Optional[Part.Shape], Optional[Part.Shape]]:
        if used_space is None or not used_space.isValid():
            return shape, shape

        trimmed = shape.copy()
        try:
            trimmed = trimmed.cut(used_space)
        except Part.OCCError as exc:
            fail(
                "Failed to subtract already used intersection volume",
                IntersectionError,
                exc
            )

        if trimmed.Volume < GEOM_EPS:
            return None, used_space

        try:
            updated_space = used_space.fuse(trimmed)
        except Part.OCCError as exc:
            fail(
                "Failed to accumulate used intersection volume",
                IntersectionError,
                exc
            )
        return trimmed, updated_space

    def _determine_type(self) -> None:
        if not self.shape:
            return
        main_axis_vec = self._axis_vector(self.primary_axis_idx)
        face_contact = False
        for edge in self.shape.Edges:
            if edge.Length < GEOM_EPS:
                continue
            v0 = edge.Vertexes[0].Point
            v1 = edge.Vertexes[1].Point
            direction = self._vec_normalized(v1 - v0)
            aligned = direction.isEqual(main_axis_vec, GEOM_EPS) or direction.isEqual(-main_axis_vec, GEOM_EPS)
            if not aligned:
                continue
            on_p1 = any(self._colinear(edge, candidate) for candidate in self.part1.Shape.Edges)
            on_p2 = any(self._colinear(edge, candidate) for candidate in self.part2.Shape.Edges)
            if on_p1 and on_p2:
                self.intersection_type = IntersectionType.CORNER
                return
            if on_p1 or on_p2:
                face_contact = True
        self.intersection_type = IntersectionType.FACE if face_contact else IntersectionType.CROSS

    def _colinear(self, edge_a: Part.Edge, edge_b: Part.Edge) -> bool:
        if len(edge_a.Vertexes) < 2 or len(edge_b.Vertexes) < 2:
            fail(
                "Edges missing vertex data for colinearity test",
                IntersectionError
            )
        p1a, p1b = edge_a.Vertexes[0].Point, edge_a.Vertexes[1].Point
        p2a, p2b = edge_b.Vertexes[0].Point, edge_b.Vertexes[1].Point
        if (p1b - p1a).Length < GEOM_EPS or (p2b - p2a).Length < GEOM_EPS:
            return False
        direction = self._vec_normalized(p1b - p1a)
        return (p2a - p1a).cross(direction).Length < GEOM_EPS and \
               (p2b - p1a).cross(direction).Length < GEOM_EPS

    @staticmethod
    def _dominant_axis_index(dimensions: List[float]) -> int:
        index = 0
        max_length = dimensions[0]
        if dimensions[1] > max_length:
            index = 1
            max_length = dimensions[1]
        if dimensions[2] > max_length:
            index = 2
        return index

    # --- Finger box generation -------------------------------------------------

    def generate_fingers(self) -> None:
        """Create finger boxes once geometry is available."""
        if not self._has_geometry or not self.bbox:
            self._finger_boxes = []
            return
        self._finger_boxes = self._create_finger_boxes()

    def is_valid_fingers(self) -> bool:
        return bool(self._finger_boxes)
        

    def _create_finger_boxes(self) -> List[Part.Shape]:
        if not self.bbox or self.longest_edge_len <= GEOM_EPS:
            return []
        finger_count = self._finger_count()
        if finger_count <= 0:
            return []
        segment_length = self.longest_edge_len / float(finger_count)
        half_kerf = self.kerf * 0.5
        axis = self.primary_axis_idx
        boxes: List[Part.Shape] = []
        for index in range(finger_count):
            start, length = self._segment_with_kerf(segment_length, index, finger_count, half_kerf)
            if length <= GEOM_EPS:
                continue
            base = self._segment_base(axis, start)
            size = self._segment_size(axis, length)
            boxes.append(Part.makeBox(size[0], size[1], size[2], base))
        return boxes

    def _finger_count(self) -> int:
        if self.intersection_type == IntersectionType.CROSS:
            return CROSS_JOINT_FINGER_COUNT
        minimum_length = max(1.0, self.finger_length)
        max_fingers = int(self.longest_edge_len / minimum_length)
        candidate = max_fingers if max_fingers % 2 == 1 else max_fingers - 1
        while candidate >= MIN_FINGER_COUNT and (self.longest_edge_len / candidate) + GEOM_EPS < minimum_length:
            candidate -= 2
        if candidate >= MIN_FINGER_COUNT:
            return candidate
        fallback = MIN_FINGER_COUNT
        actual = self.longest_edge_len / float(fallback)
        if actual + GEOM_EPS < minimum_length:
            part_label_1 = getattr(self.part1, "Label", self.part1.Name)
            part_label_2 = getattr(self.part2, "Label", self.part2.Name)
            log(
                f"Requested min finger length {minimum_length:.2f} mm reduced to {actual:.2f} mm"
                f" for intersection between '{part_label_1}' and '{part_label_2}'.",
                1
            )
        return fallback

    def _segment_with_kerf(self,
                           segment: float,
                           index: int,
                           total: int,
                           half_kerf: float) -> Tuple[float, float]:
        start = index * segment
        length = segment
        if index > 0:
            start += half_kerf
            length -= half_kerf
        if index < total - 1:
            length -= half_kerf
        return start, length

    def _segment_base(self, axis: int, start: float) -> FreeCAD.Vector:
        x0, y0, z0 = self.bbox.XMin, self.bbox.YMin, self.bbox.ZMin
        if axis == 0:
            return FreeCAD.Vector(x0 + start, y0, z0)
        if axis == 1:
            return FreeCAD.Vector(x0, y0 + start, z0)
        return FreeCAD.Vector(x0, y0, z0 + start)

    def _segment_size(self, axis: int, length: float) -> Tuple[float, float, float]:
        dx, dy, dz = self.bbox.XLength, self.bbox.YLength, self.bbox.ZLength
        if axis == 0:
            return length, dy, dz
        if axis == 1:
            return dx, length, dz
        return dx, dy, length

    # --- Cutting ---------------------------------------------------------------

    def cut_fingers(self,
                    shape1: Part.Shape,
                    shape2: Part.Shape) -> Tuple[Part.Shape, Part.Shape]:
        """Apply the precomputed finger boxes to the supplied shapes."""
        if not self._finger_boxes:
            return shape1, shape2
        cut1, cut2 = self._split_cutters(self._finger_boxes)
        new1 = BooleanCutter.apply(shape1, cut1)
        new2 = BooleanCutter.apply(shape2, cut2)
        return new1, new2

    def _split_cutters(self, boxes: List[Part.Shape]) -> Tuple[Part.Compound, Part.Compound]:
        cut1 = Part.Compound([boxes[i] for i in range(1, len(boxes), 2)])
        cut2 = Part.Compound([boxes[i] for i in range(0, len(boxes), 2)])
        return cut1, cut2

    @staticmethod
    def _axis_vector(index: int) -> FreeCAD.Vector:
        """Return a canonical axis vector for the provided axis index."""
        return Intersection._AXIS_VECTORS[int(index)]

    @staticmethod
    def _vec_normalized(vector: FreeCAD.Vector) -> FreeCAD.Vector:
        """Return a normalized copy that works across FreeCAD versions."""
        length = vector.Length
        if length <= GEOM_EPS:
            return FreeCAD.Vector(0, 0, 0)
        out = FreeCAD.Vector(vector.x, vector.y, vector.z)
        if hasattr(out, "normalize"):
            result = out.normalize()
            if result is not None:
                return result
            return out
        if hasattr(out, "normalized"):
            return out.normalized()
        return out.multiply(1.0 / length)


class BooleanCutter:
    """Helper to apply boolean cutters to shapes."""

    @staticmethod
    def apply(shape: Part.Shape, cutter: Part.Shape) -> Part.Shape:
        if cutter.isNull() or not cutter.isValid():
            return shape
        try:
            result = shape.cut(cutter)
        except Part.OCCError as exc:
            fail(
                "Boolean cut failed for shape",
                IntersectionError,
                exc
            )
        if hasattr(result, "removeSplitter"):
            try:
                result = result.removeSplitter()
            except Exception as exc:
                fail(
                    "removeSplitter failed for shape",
                    DocumentUpdateError,
                    exc
                )
        return BooleanCutter._largest_solid(result)

    @staticmethod
    def _largest_solid(shape: Part.Shape) -> Part.Shape:
        """Return the largest solid contained inside a compound results."""
        if not shape.isValid() or not shape.Solids:
            return shape
        if len(shape.Solids) == 1:
            return shape.Solids[0]
        best = shape.Solids[0]
        max_vol = float(best.Volume)
        for candidate in shape.Solids[1:]:
            volume = float(candidate.Volume)
            if volume > max_vol:
                best = candidate
                max_vol = volume
        return best


# =============================================================================
# 6) 2D projection/packing: ProjectionOrchestrator
# =============================================================================

class ProjectionOrchestrator:
    """Coordinate 2D projection layout creation for cloned parts."""

    def __init__(self,
                 container: Optional[FreeCAD.DocumentObject],
                 parts: List[FreeCAD.DocumentObject]):
        """Store references needed to populate layout objects."""
        self.container = container
        self.doc = container.Document if container is not None else FreeCAD.ActiveDocument
        self.parts = parts or []

    def generate(self, progress: Optional["ProgressDialog"] = None) -> None:
        """Create projection layouts for every detected thickness in the container."""
        log('ProjectionOrchestrator: starting projection build.', 1)
        if self.container is None:
            log('ProjectionOrchestrator: container missing; projections skipped.', 1)
            return

        if not self.parts:
            log('ProjectionOrchestrator: no solids provided for projection.', 1)
            return

        groups = self._group_parts_by_thickness(self.parts)
        if not groups:
            log('ProjectionOrchestrator: no thickness groups to process.', 1)
            return

        offset = self._starting_offset()
        local_progress = progress
        owns_progress = False
        if local_progress is None:
            local_progress = ProgressDialog(max(1, len(groups)), 'Projection Layouts')
            owns_progress = True
        else:
            local_progress.set_phase('Projection Layouts', max(1, len(groups)))

        try:
            for index, (thickness, group) in enumerate(groups):
                if local_progress:
                    local_progress.update(index, f'Projection sheet for {thickness:.1f} mm parts')
                builder = ThicknessSheetLayout(
                    container=self.container,
                    thickness=thickness,
                    parts=group,
                    offset=FreeCAD.Vector(offset),  # use a copy so later mutations don't affect placed layouts
                    progress=local_progress
                )
                height = builder.build()
                offset.y += height + LAYOUT_GAP
        finally:
            if owns_progress and local_progress:
                local_progress.finish()

        log('ProjectionOrchestrator: completed projection layouts.', 1)

    def _group_parts_by_thickness(self,
                                  parts: List[FreeCAD.DocumentObject]) -> List[Tuple[float, List[FreeCAD.DocumentObject]]]:
        """Return parts grouped by their minimum bounding box dimension (thickness)."""
        groups: Dict[float, List[FreeCAD.DocumentObject]] = {}
        for obj in parts:
            thickness = self._part_thickness(obj)
            groups.setdefault(thickness, []).append(obj)
        ordered = sorted(groups.items(), key=lambda item: item[0])
        log(f'ProjectionOrchestrator: grouped parts into {len(ordered)} thickness buckets.', 1)
        return ordered

    def _part_thickness(self, obj: FreeCAD.DocumentObject) -> float:
        """Return the rounded minimum extent of the part to treat as sheet thickness."""
        bb = obj.Shape.BoundBox
        thickness = min(bb.XLength, bb.YLength, bb.ZLength)
        return round(thickness, 1)

    def _starting_offset(self) -> FreeCAD.Vector:
        """Return the translation vector used to place layouts beside the parts."""
        overall = FreeCAD.BoundBox()
        for part in self.parts:
            overall.add(part.Shape.BoundBox)
        offset = FreeCAD.Vector(overall.XMax + LAYOUT_GAP, overall.YMin, 0.0)
        log(f'ProjectionOrchestrator: layout offset initialised to {offset}.', 2)
        return offset


class ShelfPacker:
    """Simple shelf-packing algorithm for 2D projection shapes."""

    def __init__(self, margin: float):
        self.margin = margin

    def pack(self,
             items: List[Dict[str, Any]],
             sheet_width: float) -> List[Part.Shape]:
        shelves: List[Dict[str, float]] = []
        placed_shapes: List[Part.Shape] = []
        for item in items:
            placed_shapes.append(self._place_item(item, shelves, sheet_width))
        return placed_shapes

    def _place_item(self,
                    item: Dict[str, Any],
                    shelves: List[Dict[str, float]],
                    sheet_width: float) -> Part.Shape:
        index, rotate = self._find_shelf_for_item(item, shelves, sheet_width)
        if index is None:
            rotate = self._rotation_for_new_shelf(item)
            width, height = self._dimensions(item, rotate)
            shelf = self._append_shelf(shelves, height)
        else:
            width, height = self._dimensions(item, rotate)
            shelf = shelves[index]

        shape = item['shape'].copy()
        if rotate:
            shape.rotate(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 90)

        bbox = shape.BoundBox
        move = FreeCAD.Vector(shelf['cursor'] - bbox.XMin, shelf['y'] - bbox.YMin, -bbox.ZMin)
        shape.translate(move)
        shelf['cursor'] += width + self.margin
        return shape

    def _find_shelf_for_item(self,
                             item: Dict[str, Any],
                             shelves: List[Dict[str, float]],
                             sheet_width: float) -> Tuple[Optional[int], bool]:
        """Locate the best shelf for an item, returning shelf index and rotation flag."""
        best_index: Optional[int] = None
        best_leftover = float('inf')
        best_rotate = False

        for index, shelf in enumerate(shelves):
            for rotate in (False, True):
                width, height = self._dimensions(item, rotate)
                if not self._shelf_fits(shelf, width, height, sheet_width):
                    continue
                leftover = shelf['height'] - (height + self.margin)
                if leftover < best_leftover:
                    best_leftover = leftover
                    best_index = index
                    best_rotate = rotate
        return best_index, best_rotate

    @staticmethod
    def _rotation_for_new_shelf(item: Dict[str, Any]) -> bool:
        """Return True when the item should rotate to better fit a fresh shelf."""
        return item['height'] > item['width']

    def _append_shelf(self, shelves: List[Dict[str, float]], height: float) -> Dict[str, float]:
        """Create a new shelf positioned below the previous one."""
        base_y = shelves[-1]['y'] + shelves[-1]['height'] if shelves else 0.0
        shelf = {'y': base_y, 'height': height + self.margin, 'cursor': 0.0}
        shelves.append(shelf)
        return shelf

    @staticmethod
    def _dimensions(item: Dict[str, Any], rotate: bool) -> Tuple[float, float]:
        """Return item dimensions, swapping axes when rotation is requested."""
        if rotate:
            return item['height'], item['width']
        return item['width'], item['height']

    def _shelf_fits(self,
                    shelf: Dict[str, float],
                    width: float,
                    height: float,
                    sheet_width: float) -> bool:
        """Return True when the provided item fits in the shelf."""
        if height + self.margin > shelf['height']:
            return False
        return (shelf['cursor'] + width + self.margin) <= sheet_width


class ThicknessSheetLayout:
    """Create and attach a packed projection layout for a single thickness value."""

    def __init__(self,
                 container: Optional[FreeCAD.DocumentObject],
                 thickness: float,
                 parts: List[FreeCAD.DocumentObject],
                 offset: FreeCAD.Vector,
                 progress: Optional["ProgressDialog"]):
        """Store the parameters needed to flatten and pack parts."""
        self.container = container
        self.doc = container.Document if container is not None else FreeCAD.ActiveDocument
        self.thickness = thickness
        self.parts = parts or []
        self.offset = offset
        self.progress = progress
        self.margin = PROJECTION_MARGIN
        self.tol = GEOM_EPS

    def build(self) -> float:
        """Flatten, pack, and attach a layout for the stored parts."""
        log(f'ThicknessSheetLayout: building layout for {self.thickness:.1f} mm.', 1)
        if not self.parts:
            log('ThicknessSheetLayout: no parts provided for this thickness.', 1)
            return 0.0

        items = self._collect_projections()
        if not items:
            log('ThicknessSheetLayout: projection items list empty.', 1)
            return 0.0

        ordered = self._sorted_items(items)
        compound, layout_height = self._build_layout_compound(ordered)
        if compound is None:
            return 0.0

        layout_name = self._layout_label()
        self._attach_layout_to_container(layout_name, compound)
        log(f"ThicknessSheetLayout: layout '{layout_name}' created.", 1)
        return layout_height

    def _collect_projections(self) -> List[Dict[str, Any]]:
        """Flatten each part's largest face into projection dictionaries."""
        projections: List[Dict[str, Any]] = []
        if self.progress:
            self.progress.set_phase('Generating 2D projections', len(self.parts))
        for index, obj in enumerate(self.parts):
            label = getattr(obj, 'Label', obj.Name)
            if self.progress:
                self.progress.update(index, f'Projecting: {label}')

            face = self._largest_face(obj)
            if face is None:
                log(f"ThicknessSheetLayout: skipped '{label}'; no face available.", 1)
                continue

            flat = self._project_to_xy(obj, face)
            if flat is None:
                log(f"ThicknessSheetLayout: skipped '{label}'; projection failed.", 1)
                continue

            bbox = flat.BoundBox
            if not bbox.isValid():
                log(f"ThicknessSheetLayout: skipped '{label}'; invalid bounding box.", 1)
                continue

            projections.append({
                'shape': flat,
                'label': label,
                'width': bbox.XLength,
                'height': bbox.YLength
            })

        log(f'ThicknessSheetLayout: collected {len(projections)} flattened faces.', 1)
        return projections

    @staticmethod
    def _largest_face(obj: FreeCAD.DocumentObject) -> Optional[Part.Face]:
        """Return the face with maximum area from the given object."""
        if not (hasattr(obj, 'Shape') and obj.Shape.Faces):
            return None
        return max(obj.Shape.Faces, key=ThicknessSheetLayout._face_area)

    def _project_to_xy(self, obj: FreeCAD.DocumentObject, face: Part.Face) -> Optional[Part.Face]:
        """Rotate the provided face so it lies within the XY plane."""
        if not face.Wires:
            return None

        rotated_face = self._rotated_face(obj, face)
        if rotated_face is None:
            log(f"ThicknessSheetLayout: rotation failed for '{obj.Label}'.", 1)
            return None

        flattened = self._normalized_face(rotated_face)
        if flattened.Area <= self.tol:
            log(f"ThicknessSheetLayout: rotated face has near-zero area for '{obj.Label}'.", 1)
            return None

        return flattened

    def _rotated_face(self,
                      obj: FreeCAD.DocumentObject,
                      face: Part.Face) -> Optional[Part.Face]:
        """Return a transformed copy of the face aligned with the XY plane."""
        label = getattr(obj, 'Label', obj.Name)
        try:
            rotation = self._rotation_to_xy(face)
        except ProjectionError:
            raise
        except Exception as exc:
            fail(
                f"Unexpected failure computing rotation for '{label}'",
                ProjectionError,
                exc
            )

        transformed = face.copy()
        try:
            combined = obj.Placement.multiply(FreeCAD.Placement(FreeCAD.Vector(), rotation))
        except Exception as exc:
            fail(
                f"Unable to combine placement and rotation for '{label}'",
                ProjectionError,
                exc
            )

        matrix = combined.toMatrix()

        try:
            transformed = transformed.transformShape(matrix)
        except AttributeError:
            try:
                transformed = transformed.transformGeometry(matrix)
            except Exception as exc:
                fail(
                    f"Failed to transform face into XY plane for '{label}'",
                    ProjectionError,
                    exc
                )
        except Exception as exc:
            fail(
                f"Failed to transform face into XY plane for '{label}'",
                ProjectionError,
                exc
            )

        return transformed

    def _rotation_to_xy(self, face: Part.Face) -> FreeCAD.Rotation:
        """Compute the rotation that aligns a face normal with the +Z axis."""
        try:
            normal = face.normalAt(0.5, 0.5)
        except Exception as exc:
            fail('Cannot evaluate face normal for projection', ProjectionError, exc)
        if normal.Length <= self.tol:
            fail('Degenerate face encountered during projection', ProjectionError)
        return FreeCAD.Rotation(normal, FreeCAD.Vector(0, 0, 1))

    def _normalized_face(self, face: Part.Face) -> Part.Face:
        """Translate the face so its bounding box starts at the origin."""
        bbox = face.BoundBox
        move = FreeCAD.Vector(-bbox.XMin, -bbox.YMin, -bbox.ZMin)
        face.translate(move)
        return face

    def _build_layout_compound(self,
                               items: List[Dict[str, Any]]) -> Tuple[Optional[Part.Shape], float]:
        """Create a translated compound ready for document insertion."""
        total_area = self._total_packing_area(items)
        if total_area <= 0.0:
            log('ThicknessSheetLayout: projected items have no packing area.', 1)
            return None, 0.0

        sheet_width = self._sheet_width(total_area)
        packer = ShelfPacker(self.margin)
        placed_shapes = packer.pack(items, sheet_width)
        if not placed_shapes:
            log('ThicknessSheetLayout: no shapes were placed during packing.', 1)
            return None, 0.0

        compound = Part.Compound(placed_shapes)
        if not compound.isValid():
            log('ThicknessSheetLayout: packed compound is invalid.', 1)
            return None, 0.0

        compound.translate(self.offset)
        layout_height = compound.BoundBox.YLength if compound.isValid() else 0.0
        return compound, layout_height

    def _sorted_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return items ordered by descending area to improve packing quality."""
        return sorted(items, key=self._item_area, reverse=True)

    def _total_packing_area(self, items: List[Dict[str, Any]]) -> float:
        """Return the summed area needed for all items including margins."""
        total = 0.0
        for item in items:
            total += (item['width'] + self.margin) * (item['height'] + self.margin)
        return total

    @staticmethod
    def _sheet_width(total_area: float) -> float:
        """Estimate a sheet width using a simple heuristic multiplier."""
        return math.sqrt(total_area * 1.5)

    def _layout_label(self) -> str:
        """Return a unique layout label for the stored thickness."""
        if abs(self.thickness - round(self.thickness)) < 1e-6:
            label_value = int(round(self.thickness))
        else:
            label_value = round(self.thickness, 2)
        base = f'Layout_{label_value:g}mm'
        token, index = self._next_label(base)
        return f'{token}_{index:03d}'

    def _attach_layout_to_container(self,
                                    layout_name: str,
                                    compound: Part.Shape) -> None:
        """Insert the packed layout into the cloned container."""
        doc = self.doc
        if doc is None:
            fail('Document is unavailable for layout insertion', DocumentUpdateError)

        base_name = self._safe_label(layout_name)
        unique_name = base_name
        suffix = 1
        while doc.getObject(unique_name):
            unique_name = f'{base_name}_{suffix}'
            suffix += 1
        try:
            obj = doc.addObject('Part::Feature', unique_name)
        except Exception as exc:
            fail(
                f"Failed to add layout '{layout_name}' to the document",
                DocumentUpdateError,
                exc
            )

        obj.Shape = compound
        obj.Label = layout_name
        if self.container is not None and hasattr(self.container, 'addObject'):
            try:
                self.container.addObject(obj)
            except Exception as exc:
                fail(
                    f"Failed to attach layout '{layout_name}' to container '{self.container.Label}'",
                    DocumentUpdateError,
                    exc
                )

    @staticmethod
    def _face_area(face: Part.Face) -> float:
        """Return the area of a face, guarding against missing data."""
        if not face.isValid():
            fail('Invalid face encountered while computing area', ProjectionError)
        return float(face.Area)

    @staticmethod
    def _safe_label(text: str) -> str:
        token = re.sub(r'[^0-9A-Za-z]+', '_', (text or '')).strip('_')
        return token or 'Object'

    def _next_label(self, base_label: str) -> Tuple[str, int]:
        doc = self.doc
        token = self._safe_label(base_label)
        prefix = f"{token}_"
        max_index = 0
        for obj in doc.Objects:
            label = getattr(obj, 'Label', '')
            if not label.startswith(prefix):
                continue
            suffix = label[len(prefix):]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
        return token, max_index + 1

    @staticmethod
    def _item_area(item: Dict[str, Any]) -> float:
        """Return the plan area for a projection item."""
        return item['width'] * item['height']

# =============================================================================
# 7) Progress helper (console-only status, optional progressbar)
# =============================================================================

class ProgressDialog:
    """Progress dialog managed explicitly and reusable across phases."""

    def __init__(self, max_value: int = 100, title: str = "Processing..."):
        self.dialog = QtWidgets.QProgressDialog(title, "Cancel", 0, max_value)
        self.dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.dialog.setAutoClose(False)
        self.dialog.setAutoReset(False)
        self.dialog.setMinimumDuration(0)
        self.dialog.setValue(0)
        self.dialog.show()

    def __enter__(self) -> "ProgressDialog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()

    def set_phase(self, title: str, max_value: int) -> None:
        """Change title and reset value for a new phase."""
        if not self.dialog:
            log(f"[{title}] Starting phase...", 1)
            return
        self.dialog.setLabelText(title)
        self.dialog.setMaximum(max_value)
        self.dialog.setValue(0)
        self.process_events()

    def update(self, value: int, text: str = "") -> None:
        """Advance the progress dialog and optionally display a status message."""
        if text:
            log(text, 1)

        if not self.dialog:
            return

        self.dialog.setValue(int(value))
        if self.dialog.wasCanceled():
            raise InterruptedError()

    def finish(self) -> None:
        """Close the dialog."""
        if self.dialog:
            self.dialog.close()
        log("Progress scope finished.", 1)

    def process_events(self) -> None:
        """Flush Qt events so the dialog keeps responding."""
        QtWidgets.QApplication.processEvents()


# =============================================================================
# 8) Orchestrator: FingerJointOrchestrator (selection → projection)
# =============================================================================

class FingerJointOrchestrator:
    """Top-level coordinator: selection → duplication → cutting → projection."""

    def __init__(self):
        self.doc: Optional[FreeCAD.Document] = None
        self.container: Optional[FreeCAD.DocumentObject] = None
        self.clone_container: Optional[FreeCAD.DocumentObject] = None
        self.used_space: Optional[Part.Shape] = None
        self._cloned_parts: List[FreeCAD.DocumentObject] = []
        self._globals: Optional[GlobalSettings] = None
        self.progress: Optional[ProgressDialog] = None

    def run(self) -> None:
        """Execute selection → duplication → cutting → projection."""

        self._set_active_container()
        self._collect_user_preferences()
        if self._globals is None:
            log("User canceled — nothing to do.", 1)
            return

        self._clone_container()
        with ProgressDialog(100, "Finger Joint Processing") as progress:
            self.progress = progress
            self._process_finger_joints_and_apply()
            self._generate_projections()
            self.progress = None

    def _process_finger_joints_and_apply(self) -> None:
        """Compute finger joints, apply them to clones, and handle UX/progress."""
        if not self._cloned_parts:
            fail("No parts remain eligible for processing after duplication.")

        if self.progress is None:
            fail("Progress dialog not initialised for finger joint processing.", ProgressError)

        if self._globals is None:
            fail("Global settings missing for finger joint computation.", ProgressError)

        try:
            total_pairs = (len(self._cloned_parts) * (len(self._cloned_parts) - 1)) // 2
            log(f"Processing {len(self._cloned_parts)} parts → {total_pairs} intersections.", 1)
            self.progress.set_phase("Applying Finger Joints", max(1, total_pairs))

            pair_index = 0
            for i, part_a in enumerate(self._cloned_parts):
                for j in range(i + 1, len(self._cloned_parts)):
                    part_b = self._cloned_parts[j]
                    self.progress.update(pair_index, f"Pair: {part_a.Label} vs {part_b.Label}")
                    pair_index += 1
                    intersection = Intersection(part_a, part_b, self._globals, self.used_space)
                    self.used_space = intersection.process()

            self.progress.update(100, "✓ All tasks completed.")
            log("✓ Finger joint processing completed.", 1)
            if hasattr(FreeCADGui, "Selection"):
                FreeCADGui.Selection.clearSelection()
            try:
                self.doc.recompute()
            except Exception as exc:
                fail(
                    "Document recompute failed",
                    DocumentUpdateError,
                    exc
                )
        except InterruptedError as exc:
            log("Processing cancelled by user", level='warning')
            raise UserCancelledError(str(exc)) from exc
        except Exception as exc:
            traceback.print_exc()
            if isinstance(exc, FingerJointError):
                raise
            fail(f"Unexpected error: {exc}")


    def _generate_projections(self) -> None:
        if not self.clone_container or not self._cloned_parts:
            return
        ProjectionOrchestrator(self.clone_container, self._cloned_parts).generate(self.progress)


    def _set_active_container(self) -> FreeCAD.DocumentObject:
        selection_api = getattr(FreeCADGui, "Selection", None)
        if selection_api is None:
            fail("FreeCADGui selection API unavailable.")
        selection = selection_api.getSelection()
        if not selection:
            fail("Select a Part/App::Part container before running the macro.")
        self.container = selection[0]
        type_id = getattr(self.container, "TypeId", "")
        if type_id not in {"App::Part", "Part::Part"}:
            fail("Selected object is not a Part container.")     
        container_doc = getattr(self.container, 'Document', None)
        if container_doc is None:
            fail("Selected container is not attached to a document.")
        self.doc = container_doc



    def _collect_user_preferences(self) -> None:
        """Display the streamlined dialog and store global settings."""
        parent_accessor = getattr(FreeCADGui, "getMainWindow", None)
        if parent_accessor is None or not callable(parent_accessor):
            fail("FreeCAD main window accessor is unavailable")
        try:
            parent = parent_accessor()
        except Exception as exc:
            fail("Failed to obtain FreeCAD main window", DocumentUpdateError, exc)

        dialog = FingerJointDialog(self.doc, parent)
        values = dialog.get_values()
        if values:
            log(
                f"Dialog confirmed: kerf={values.kerf:.3f} mm, "
                f"finger length={values.finger_length:.2f} mm.",
                1
            )
            self._globals = values
        else:
            log("Dialog dismissed with no values.", 1)
            self._globals = None

    def _clone_container(self) -> None:
        """Create a new App::Part populated with clones from the original container."""
        base_label = getattr(self.container, "Label", self.container.Name)
        candidate_name = self._unique_name(base_label)
        try:
            clone_container = self.doc.addObject("App::Part", candidate_name)
        except Exception as exc:
            fail(
                f"Failed to create cloned container for '{base_label}'",
                DocumentUpdateError,
                exc
            )
        clone_container.Label = candidate_name
        self.clone_container = clone_container

        clones: List[FreeCAD.DocumentObject] = []
        visited: Set[int] = set()
        stack = list(reversed(getattr(self.container, "Group", []) or []))
        while stack:
            node = stack.pop()
            ident = id(node)
            if ident in visited:
                continue
            visited.add(ident)

            children = getattr(node, "Group", []) or []
            if children:
                stack.extend(reversed(children))

            clone = self._clone_part(node)
            if clone is None:
                continue
            try:
                clone_container.addObject(clone)
            except Exception as exc:
                fail(
                    f"Failed to add clone '{clone.Label}' to '{clone_container.Label}'",
                    DocumentUpdateError,
                    exc
                )
            clones.append(clone)

        if not clones:
            fail("Selected container has no processable solids.")

        self._cloned_parts = clones
        if hasattr(clone_container, "ViewObject"):
            clone_container.ViewObject.Visibility = True
        if hasattr(self.container, "ViewObject"):
            self.container.ViewObject.Visibility = False

    def _clone_part(self, obj: FreeCAD.DocumentObject) -> Optional[FreeCAD.DocumentObject]:
        """Create a Part::Feature clone with copied placement and shape."""
        if not hasattr(obj, "ViewObject") or not getattr(obj.ViewObject, "Visibility", True):
            return None
        if not hasattr(obj, "Shape") or not isinstance(obj.Shape, Part.Shape):
            return None
        if not obj.Shape.isValid() or obj.Shape.Volume <= 0:
            return None
        bbox = obj.Shape.BoundBox
        if min(bbox.XLength, bbox.YLength, bbox.ZLength) <= GEOM_EPS:
            return None
        type_id = getattr(obj, "TypeId", "")
        if not type_id.startswith("Part::"):
            return None
        base_label = getattr(obj, "Label", obj.Name)
        name_candidate = self._unique_name(base_label)
        try:
            clone = self.doc.addObject("Part::Feature", name_candidate)
        except Exception as exc:
            fail(
                f"Failed to create clone for '{obj.Label}'",
                DocumentUpdateError,
                exc
            )
        clone.Label = name_candidate
        clone.Placement = obj.Placement
        clone.Shape = obj.Shape.copy()
        if hasattr(obj, "ViewObject") and hasattr(clone, "ViewObject"):
            clone.ViewObject.ShapeColor = getattr(obj.ViewObject, "ShapeColor", clone.ViewObject.ShapeColor)
        return clone

    def _unique_name(self, base_label: str) -> str:
        token = re.sub(r'[^0-9A-Za-z]+', '_', (base_label or '')).strip('_') or 'Object'
        prefix = f"{token}_"
        max_index = 0
        for obj in self.doc.Objects:
            label = getattr(obj, 'Label', '')
            if not label.startswith(prefix):
                continue
            suffix = label[len(prefix):]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
        base_name = f"{token}_{(max_index + 1):03d}"
        candidate = base_name
        suffix = 1
        while self.doc.getObject(candidate):
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        return candidate


# =============================================================================
# 9) Main entry
# =============================================================================


def main() -> None:
    """Execute the container processor with structured logging and guards."""
    log("Starting Container Processor…", 1)
    try:
        FingerJointOrchestrator().run()
    except UserCancelledError as exc:
        log(f"Processing cancelled: {exc}", 'info')
    except FingerJointError as exc:
        log(f"Finger joint processing failed: {exc}", 'warning')
        raise
    except Exception as exc:
        traceback.print_exc()
        fail(
            f"Unexpected unhandled exception: {exc}",
            FingerJointError,
            exc
        )


# Standard entry point for FreeCAD macros.
if __name__ == "__main__":
    main()
