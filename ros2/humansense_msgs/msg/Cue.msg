# A non-verbal cue attributed to one person.
uint8 SEVERITY_INFO=0
uint8 SEVERITY_ATTENTION=1
uint8 SEVERITY_URGENT=2

int32 person_id
string kind              # fall | fall_risk | raised_hand | protective_posture | arms_crossed | seated
string label             # human-readable description
uint8 severity
