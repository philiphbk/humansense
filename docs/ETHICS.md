# Responsible use

HumanSense observes people in a care setting. Please read this before deploying it.

* **Not a medical device and not a safety function.** It is a research prototype. Its fall cue
  misses falls and raises false alarms (see the benchmark results). It must never be the only
  safeguard for a patient or the only thing standing between a robot and a person. Keep the robot's
  own certified collision avoidance and e-stop in charge.
* **Privacy by design.** Everything runs locally; no frame leaves the machine. The ROS node
  publishes *derived* data (boxes, cue labels, a risk score) and publishes video only if
  `publish_annotated` is on. `utils/privacy.blur_heads` can pixelate heads in any output you record.
  Events are stored in a local SQLite file; set `HUMANSENSE_EVENTS_DB` to place it on encrypted storage.
* **Emotion recognition is opt-in.** It is the least reliable module (see the FER2013 results, and
  accuracy drops further with a mask), it is trained on a research-only dataset, and inferring
  emotion is regulated in some jurisdictions (for example the EU AI Act restricts it in workplace
  and education settings and treats it as high-risk elsewhere). It is therefore excluded from the
  ROS node's default modules and carries a low weight in the risk score. Enable it only with a clear
  purpose, legal review and, where required, consent.
* **Bias.** The detectors and the emotion model were trained on public datasets that are not
  representative of every age, skin tone, body shape, mobility aid or clinical posture (patients in
  wheelchairs, on gurneys, or using walkers are poorly covered). Validate on your own site before
  relying on any output, and re-validate when the camera or environment changes.
* **Consent and signage.** Cameras in clinical spaces usually require notice and a lawful basis.
  Involve your data-protection officer and ethics board.
