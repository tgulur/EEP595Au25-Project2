#!/usr/bin/env bash
#
# run_tests.sh - Test runner script for CAN IDS project
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}                    CAN IDS PROJECT TEST SUITE${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not found${NC}"
    echo "Please run: python -m venv .venv && source .venv/bin/activate"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}✓${NC} Activating virtual environment..."
source .venv/bin/activate

# Install test dependencies if pytest not found
if ! python -c "import pytest" 2>/dev/null; then
    echo -e "${YELLOW}⚠${NC}  Installing test dependencies..."
    pip install -q pytest pytest-cov pytest-xdist
fi

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}                        RUNNING TESTS${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Parse command line arguments
TEST_TYPE="${1:-all}"
TEST_MARKERS=""
TEST_ARGS="-v --tb=short"

case "$TEST_TYPE" in
    unit)
        echo -e "${GREEN}Running unit tests only...${NC}"
        TEST_MARKERS="-m unit"
        ;;
    integration)
        echo -e "${GREEN}Running integration tests only...${NC}"
        TEST_MARKERS="-m integration"
        ;;
    visualization)
        echo -e "${GREEN}Running visualization tests only...${NC}"
        TEST_MARKERS="-m visualization"
        ;;
    edge)
        echo -e "${GREEN}Running edge case tests only...${NC}"
        TEST_MARKERS="-m edge_case"
        ;;
    fast)
        echo -e "${GREEN}Running fast tests (excluding slow)...${NC}"
        TEST_MARKERS="-m 'not slow'"
        ;;
    coverage)
        echo -e "${GREEN}Running tests with coverage report...${NC}"
        TEST_ARGS="$TEST_ARGS --cov=src --cov-report=html --cov-report=term"
        ;;
    parallel)
        echo -e "${GREEN}Running tests in parallel...${NC}"
        TEST_ARGS="$TEST_ARGS -n auto"
        ;;
    all)
        echo -e "${GREEN}Running all tests...${NC}"
        ;;
    *)
        echo -e "${RED}✗${NC} Unknown test type: $TEST_TYPE"
        echo ""
        echo "Usage: $0 [test_type]"
        echo ""
        echo "Available test types:"
        echo "  all          - Run all tests (default)"
        echo "  unit         - Run unit tests only"
        echo "  integration  - Run integration tests only"
        echo "  visualization - Run visualization tests only"
        echo "  edge         - Run edge case tests only"
        echo "  fast         - Run fast tests (exclude slow tests)"
        echo "  coverage     - Run with coverage report"
        echo "  parallel     - Run tests in parallel"
        echo ""
        exit 1
        ;;
esac

# Run pytest
echo ""
if pytest $TEST_ARGS $TEST_MARKERS tests/; then
    echo ""
    echo -e "${GREEN}======================================================================${NC}"
    echo -e "${GREEN}                       ALL TESTS PASSED ✓${NC}"
    echo -e "${GREEN}======================================================================${NC}"
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}======================================================================${NC}"
    echo -e "${RED}                       TESTS FAILED ✗${NC}"
    echo -e "${RED}======================================================================${NC}"
    echo ""
    exit 1
fi
