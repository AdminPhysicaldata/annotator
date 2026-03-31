#!/bin/bash
# VIVE Labeler — Quick Launch Script

set -e

SESSION_DIR="${1:-data/session_20260216_223719}"

echo "========================================"
echo "   VIVE LABELER — Multi-Camera Annotation"
echo "========================================"
echo ""
echo "Session: $SESSION_DIR"
echo ""
echo "Keyboard shortcuts:"
echo "  Space         Play/Pause"
echo "  Left/Right    Previous/Next frame"
echo "  Shift+L/R     Skip 10 frames"
echo "  I             Set IN point"
echo "  O             Set OUT point"
echo "  Ctrl+S        Save annotations"
echo "  Ctrl+O        Open session"
echo "========================================"
echo ""

python -m src.main "$SESSION_DIR"
