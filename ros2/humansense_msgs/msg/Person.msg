# A tracked person in the camera image (pixel coordinates of the ORIGINAL image).
int32 id                 # stable track id
int32 x1
int32 y1
int32 x2
int32 y2
float32 confidence
float32 speed            # body heights moved per processed frame
uint32 still_frames      # consecutive processed frames with almost no movement
