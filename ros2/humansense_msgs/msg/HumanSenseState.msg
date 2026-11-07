# Everything HumanSense perceived in one camera frame.
# header.stamp / frame_id are copied from the input image, so (now - header.stamp) is the true
# sensing latency.
std_msgs/Header header
uint32 frame_id               # HumanSense's own processed-frame counter
uint32 image_width
uint32 image_height

humansense_msgs/Person[] people
uint32 person_count
string density_level          # low | medium | high
string proximity_level        # none | caution | warning
humansense_msgs/Violation[] violations
humansense_msgs/Cue[] cues
humansense_msgs/Risk risk

float32 processing_ms         # compute time spent on this frame
