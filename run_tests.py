from __future__ import annotations

import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import tempfile
import re
import xml.etree.ElementTree as ET

try:
    import FreeCAD
    import FreeCADGui
    import importSVG
    from PySide6 import QtCore
    import box
except ImportError:
    FreeCAD = None
    FreeCADGui = None
    importSVG = None
    QtCore = None
    box = None


CONTAINER_LABEL = 'Container'
LOG_PATH = Path('run_cross.log')
TESTS_DIR = Path('tests')
EXPORT_ARCHIVE_DIR = Path('verification_exports')


@dataclass
class TestCase:
    """Describe a single FCStd test document and its reference SVG outputs."""
    name: str
    doc_path: Path
    reference_svgs: List[Path]


@dataclass
class TestResult:
    """Capture the outcome of a single automated test execution."""
    name: str
    success: bool
    details: str


def init_log() -> None:
    """Reset the log file at startup."""
    LOG_PATH.write_text('', encoding='utf-8')
    EXPORT_ARCHIVE_DIR.mkdir(exist_ok=True)


def log_action(message: str) -> None:
    """Append a timestamped log entry to the file and FreeCAD console."""
    timestamp = datetime.now().isoformat(timespec='seconds')
    line = f"{timestamp} {message}"
    with LOG_PATH.open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')
    FreeCAD.Console.PrintMessage(line + '\n')


def discover_test_cases() -> List[TestCase]:
    """Return every FCStd test case and its reference SVG outputs."""
    log_action(f"Discovering test cases under {TESTS_DIR}.")
    if not TESTS_DIR.exists():
        raise FileNotFoundError(f"Tests directory missing: {TESTS_DIR}")
    fcstd_files = sorted(TESTS_DIR.glob('*.FCStd'))
    if not fcstd_files:
        raise ValueError(f"No FCStd files found in {TESTS_DIR}.")
    cases: List[TestCase] = []
    for fcstd_path in fcstd_files:
        references = _reference_svgs_for(fcstd_path)
        case = TestCase(name=fcstd_path.stem, doc_path=fcstd_path, reference_svgs=references)
        cases.append(case)
        log_action(
            f"Discovered test '{case.name}' with {len(case.reference_svgs)} reference SVG files."
        )
    return cases


def _reference_svgs_for(doc_path: Path) -> List[Path]:
    """Return SVG files matching the FCStd's uppercase prefix."""
    prefix_upper = doc_path.stem.upper()
    svg_files = sorted(doc_path.parent.glob('*.svg'))
    matches: List[Path] = []
    for svg_path in svg_files:
        if not svg_path.stem:
            continue
        if not svg_path.stem.upper().startswith(f"{prefix_upper}_"):
            continue
        matches.append(svg_path)
    return matches


def open_document(path: Path) -> FreeCAD.Document:
    """Open the provided FCStd file and return its document."""
    log_action(f"Opening document: {path}.")
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    doc = FreeCAD.openDocument(str(path))
    FreeCAD.ActiveDocument = doc
    log_action(f"Document '{doc.Label}' opened.")
    return doc


def resolve_container(doc: FreeCAD.Document, label: str) -> FreeCAD.DocumentObject:
    """Return the requested container or fall back to the first Part/App::Part."""
    log_action(f"Searching for container labeled '{label}'.")
    fallback = None
    for obj in doc.Objects:
        type_id = getattr(obj, 'TypeId', '')
        if type_id in {"App::Part", "Part::Part"} and fallback is None:
            fallback = obj
        current_label = getattr(obj, 'Label', '')
        if current_label == label or obj.Name == label:
            log_action(f"Container '{label}' resolved (Name={obj.Name}).")
            return obj
    if fallback is None:
        raise ValueError(
            f"Container '{label}' not found and no Part/App::Part available in '{doc.Label}'."
        )
    log_action(
        f"Container '{label}' not found; using first Part/App::Part (Label={fallback.Label})."
    )
    return fallback


def build_processor(doc: FreeCAD.Document,
                    container: FreeCAD.DocumentObject) -> 'box.FingerJointOrchestrator':
    """Instantiate the orchestrator with default global settings."""
    log_action("Instantiating FingerJointOrchestrator with defaults.")
    processor = box.FingerJointOrchestrator()
    processor.doc = doc
    processor.container = container
    processor._globals = box.GlobalSettings(
        kerf=box.DEFAULT_KERF_MM,
        finger_length=box.DEFAULT_MIN_FINGER_LENGTH
    )
    return processor


def run_processor(processor: 'box.FingerJointOrchestrator') -> None:
    """Execute cloning, finger joints, and projections in order without user UI."""
    log_action("Running processor stages: clone → joints → projections.")
    processor._clone_container()
    with box.ProgressDialog(100, "Finger Joint Processing") as progress:
        processor.progress = progress
        processor._process_finger_joints_and_apply()
        processor._generate_projections()
        processor.progress = None
    log_action("Processor stages completed.")


def run_test_case(case: TestCase) -> TestResult:
    """Execute the macro pipeline and verification for a single test case."""
    log_action(
        f"Running test '{case.name}' using document '{case.doc_path}'."
    )
    references = list(case.reference_svgs)
    if references:
        log_action(
            f"Found {len(references)} reference SVG files for '{case.name}'."
        )
    else:
        log_action(
            f"No reference SVG files found for '{case.name}'; generating new baselines."
        )
    doc: Optional[FreeCAD.Document] = None
    try:
        doc = open_document(case.doc_path)
        container = resolve_container(doc, CONTAINER_LABEL)
        processor = build_processor(doc, container)
        run_processor(processor)
        if references:
            case.reference_svgs = references
            verify_test_case_layouts(doc, case)
            message = f"Verified {len(references)} reference layouts."
        else:
            created_paths = create_reference_layouts(doc, case)
            case.reference_svgs = created_paths
            message = f"Created {len(created_paths)} reference layouts for future tests."
        log_action(f"Test '{case.name}' completed successfully.")
        return TestResult(name=case.name, success=True, details=message)
    except Exception as exc:
        log_action(f"Test '{case.name}' failed: {exc}")
        traceback.print_exc()
        return TestResult(name=case.name, success=False, details=str(exc))
    finally:
        cleanup(doc)


def verify_test_case_layouts(doc: FreeCAD.Document, case: TestCase) -> None:
    """Compare every generated layout against the test's reference SVG files."""
    base_upper = case.name.upper()
    for reference_path in case.reference_svgs:
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference SVG missing: {reference_path}")
        layout_tag = _layout_tag_from_reference(reference_path, base_upper)
        log_action(
            f"Verifying test '{case.name}' layout '{layout_tag}' using reference {reference_path}."
        )
        layout_obj = _select_layout_by_tag(doc, layout_tag)
        exported_path = _export_layout_to_svg(layout_obj)
        _compare_svg_files(reference_path, exported_path)
        archived_name = f"{case.name}_{reference_path.name}"
        archived_path = EXPORT_ARCHIVE_DIR / archived_name
        shutil.copy2(exported_path, archived_path)
        log_action(
            f"Test '{case.name}' layout '{layout_obj.Label}' matches reference '{reference_path.name}'."
        )
        log_action(f"Archived verification SVG at {archived_path}.")


def create_reference_layouts(doc: FreeCAD.Document, case: TestCase) -> List[Path]:
    """Export generated layouts and persist them as new reference SVG files."""
    layout_objects = _list_generated_layouts(doc)
    if not layout_objects:
        raise ValueError(
            f"No layout objects were generated for '{case.name}'; cannot create references."
        )
    created_paths: List[Path] = []
    for layout_obj in layout_objects:
        exported_path = _export_layout_to_svg(layout_obj)
        reference_path = _reference_path_for_layout(case, layout_obj.Label)
        shutil.copy2(exported_path, reference_path)
        created_paths.append(reference_path)
        log_action(
            f"Stored new reference SVG {reference_path} from layout '{layout_obj.Label}'."
        )
    return created_paths


def _list_generated_layouts(doc: FreeCAD.Document) -> List[FreeCAD.DocumentObject]:
    """Return every document object whose label looks like an exported layout."""
    layouts: List[FreeCAD.DocumentObject] = []
    for obj in doc.Objects:
        label = getattr(obj, 'Label', '')
        if not label or 'Layout_' not in label:
            continue
        layouts.append(obj)
    layouts.sort(key=lambda candidate: getattr(candidate, 'Label', ''))
    return layouts


def _reference_path_for_layout(case: TestCase, layout_label: str) -> Path:
    """Return the filesystem path for storing a layout's reference SVG."""
    safe_label = layout_label.replace(' ', '_')
    file_name = f"{case.name.upper()}_{safe_label}.svg"
    return case.doc_path.parent / file_name


def log_test_summary(results: List[TestResult]) -> None:
    """Log a concise summary of every test case outcome."""
    log_action("Automated test summary start.")
    for result in results:
        status = 'SUCCESS' if result.success else 'FAILURE'
        log_action(f"Test '{result.name}': {status} — {result.details}")
    successes = sum(1 for result in results if result.success)
    failures = len(results) - successes
    log_action(
        f"Automated test summary end. Total={len(results)} success={successes} failure={failures}."
    )


def _select_layout_by_tag(doc: FreeCAD.Document, tag: str) -> FreeCAD.DocumentObject:
    """Return the newest layout whose label contains the provided tag."""
    matches: List[FreeCAD.DocumentObject] = []
    for obj in doc.Objects:
        label = getattr(obj, 'Label', '')
        if tag in label:
            matches.append(obj)
    if not matches:
        raise ValueError(f"No layout labels contain '{tag}'.")
    matches.sort(key=_layout_index)
    chosen = matches[-1]
    log_action(f"Selected layout '{chosen.Label}' for tag '{tag}'.")
    return chosen


def _layout_tag_from_reference(reference_path: Path, base_upper: str) -> str:
    """Return the layout tag encoded in the reference SVG file name."""
    stem = reference_path.stem
    if '_' not in stem:
        raise ValueError(f"Reference SVG '{reference_path.name}' lacks a thickness suffix.")
    prefix, suffix = stem.split('_', 1)
    if prefix.upper() != base_upper:
        raise ValueError(
            f"Reference SVG '{reference_path.name}' prefix '{prefix}' does not match test '{base_upper}'."
        )
    if not suffix:
        raise ValueError(f"Reference SVG '{reference_path.name}' suffix is empty.")
    lowered = suffix.lower()
    if lowered.startswith('layout_'):
        return suffix
    if lowered.endswith('mm'):
        return f"Layout_{suffix}"
    if lowered.replace('.', '', 1).isdigit():
        return f"Layout_{suffix}mm"
    return f"Layout_{suffix}"


def _layout_index(obj: FreeCAD.DocumentObject) -> int:
    """Return the numeric suffix from the layout label for ordering."""
    label = getattr(obj, 'Label', '')
    match = re.search(r'_([0-9]+)$', label)
    if match is None:
        return 0
    return int(match.group(1))


def _export_layout_to_svg(obj: FreeCAD.DocumentObject) -> Path:
    """Export the layout object to a temporary SVG file."""
    temp_file = tempfile.NamedTemporaryFile('w', delete=False, suffix='.svg', prefix='layout_export_', dir='/tmp')
    temp_file.close()
    export_path = Path(temp_file.name)
    doc = getattr(obj, 'Document', None)
    if doc is not None:
        try:
            doc.recompute()
        except Exception as exc:
            log_action(f"Document recompute failed before export: {exc}")
    importSVG.export([obj], str(export_path))
    if not export_path.exists() or export_path.stat().st_size == 0:
        raise ValueError(
            f"Exported SVG for '{obj.Label}' is empty at {export_path}."
            " Ensure the import-export modules support SVG without GUI."
        )
    log_action(f"Exported layout '{obj.Label}' to {export_path}.")
    return export_path


def _compare_svg_files(reference_path: Path, candidate_path: Path) -> None:
    """Compare key geometric data between two SVG files."""
    ref_root = ET.parse(reference_path).getroot()
    cand_root = ET.parse(candidate_path).getroot()
    _compare_svg_attributes(ref_root, cand_root)
    ref_paths = _collect_paths(ref_root)
    cand_paths = _collect_paths(cand_root)
    if len(ref_paths) != len(cand_paths):
        raise ValueError(
            f"Path count mismatch: reference={len(ref_paths)} candidate={len(cand_paths)}."
        )
    for index, (ref_item, cand_item) in enumerate(zip(ref_paths, cand_paths), start=1):
        if ref_item != cand_item:
            raise ValueError(
                "Path geometry mismatch:"
                f" reference={ref_item}\n"
                f"candidate={cand_item}\n"
                f"index={index}"
            )


def _compare_svg_attributes(ref_root: ET.Element, cand_root: ET.Element) -> None:
    """Ensure top-level SVG attributes match between files."""
    keys = {'width', 'height', 'viewBox'}
    for key in keys:
        ref_val = ref_root.attrib.get(key)
        cand_val = cand_root.attrib.get(key)
        if ref_val != cand_val:
            raise ValueError(
                f"SVG attribute '{key}' mismatch: reference={ref_val} candidate={cand_val}."
            )


def _collect_paths(root: ET.Element) -> List[str]:
    """Return normalized path commands sorted for deterministic comparison."""
    namespace_strip = lambda tag: tag.split('}', 1)[-1]
    paths: List[str] = []
    for element in root.iter():
        if namespace_strip(element.tag) != 'path':
            continue
        descriptor = _normalize_path_data(element.attrib.get('d', ''))
        paths.append(descriptor)
    paths.sort()
    return paths


def _normalize_path_data(data: str) -> str:
    """Normalize SVG path data for stable comparisons."""
    if not data:
        return ''
    normalized_tokens = []
    for token in data.replace(',', ' ').split():
        normalized_tokens.append(token)
    return ' '.join(normalized_tokens)


def cleanup(doc: Optional[FreeCAD.Document], close_gui: bool = False) -> None:
    """Close the provided document and optionally shut down the GUI."""
    if doc:
        log_action(
            f"Closing document '{doc.Label}' without saving to preserve the on-disk file."
        )
        FreeCAD.closeDocument(doc.Name)
    if not close_gui:
        return
    main_window = getattr(FreeCADGui, 'getMainWindow', None)
    if callable(main_window):
        window = main_window()
        if window:
            window.close()
    QtCore.QCoreApplication.quit()


def run_macro() -> None:
    """Orchestrate the automated macro execution."""
    log_action("Automation timer triggered.")
    try:
        cases = discover_test_cases()
        results: List[TestResult] = []
        for case in cases:
            result = run_test_case(case)
            results.append(result)
        log_test_summary(results)
        if any(not result.success for result in results):
            raise RuntimeError("One or more automated tests failed; see summary above.")
        log_action("Automation run completed successfully for all test cases.")
    except Exception as exc:
        log_action(f"Automation failed: {exc}")
        traceback.print_exc()
    finally:
        cleanup(None, close_gui=True)


def _running_inside_freecad() -> bool:
    """Return True when the FreeCAD modules are available."""
    return all(module is not None for module in (FreeCAD, FreeCADGui, QtCore, box))


def _launch_freecad_headless() -> None:
    """Spawn FreeCAD via Flatpak inside a virtual display and run this script."""
    script_path = Path(__file__).resolve()
    server_num = '99'
    display_value = f':{server_num}'
    command = [
        "xvfb-run",
        f"--server-num={server_num}",
        "-s",
        "-screen 0 1920x1080x24",
        "flatpak",
        "run",
        f"--env=DISPLAY={display_value}",
        "org.freecad.FreeCAD",
        str(script_path)
    ]
    print('Launching FreeCAD headlessly:', ' '.join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        sys.exit(completed.returncode)


def _schedule_freecad_run() -> None:
    """Kick off the automated run inside a FreeCAD session."""
    init_log()
    QtCore.QTimer.singleShot(0, run_macro)


if _running_inside_freecad():
    _schedule_freecad_run()
elif __name__ == '__main__':
    _launch_freecad_headless()
