"""
Demo Configuration for CV Service
This allows you to specify which writers are available for the demo.
"""

# Specify which writer IDs to use for the demo (IDs should be 1-based)
# For example, if you have samples for writers 1, 3, and 7, set:
# DEMO_WRITERS = [1, 3, 7]
# Change these to match the writers you have samples for
DEMO_WRITERS = [1, 3, 5, 7]

# Confidence threshold for the demo (can be lower for demo purposes)
DEMO_CONFIDENCE_THRESHOLD = 0.60  # Lower threshold for demo

# Mapping for display names (optional)
WRITER_DISPLAY_NAMES = {
    1: "Writer A",
    2: "Writer B",
    3: "Writer C",
    4: "Writer D",
    5: "Writer E",
    6: "Writer F",
    7: "Writer G",
    8: "Writer H",
    9: "Writer I",
    10: "Writer J",
    11: "Writer K",
    12: "Writer L",
    13: "Writer M",
}
