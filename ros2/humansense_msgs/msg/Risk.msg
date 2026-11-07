# Fused 0-100 risk score for the scene. Intended as the single signal a planner or costmap
# layer needs: slow down / yield at ELEVATED, stop and call for help at CRITICAL.
uint8 LEVEL_OK=0
uint8 LEVEL_ELEVATED=1
uint8 LEVEL_CRITICAL=2

uint8 level
uint8 score
string[] reasons
