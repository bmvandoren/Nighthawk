# Giving model feedback (correcting Nighthawk's mistakes)

When Nighthawk makes an incorrect classification, submitting feedback will help improve the model's performance in the future. This document outlines the steps for submitting feedback on Nighthawk annotations.

## Choose your preferred audio program to view detailed instructions:

[View Raven Pro feedback instructions](model_feedback_raven.md).

[View Audacity feedback instructions](model_feedback_audacity.md).

## FAQ

### Do I need to review an entire file in order to submit feedback?

No. Only entries in the `category_update` column will be incorporated into further model training. Any unreviewed detections will have blank entries in `category_update` and will be ignored.

### What if I only want to review for false positives, and not make ID corrections?

No problem! Feel free to only enter `n` in the `category_update` column where Nighthawk has returned a false positive, and leave everything else blank. All other unreviewed detections will have blank entries in `category_update` and will be ignored.

### Should I add identifications for vocalizations that are not flight calls?

No. Vocalizations that are not flight calls should be ignored (or marked as `n` if they triggered a Nighthawk detection), with the exception of flight songs (see below).

### What about flight songs (i.e. birds singing in flight)?

Songs given by migrating birds in flight should be coded as flight calls because they are vocalizations given by birds on the move. For example, Scarlet Tanagers frequently sing during migratory flights. If you are not sure whether a song was given in flight, look at the spectrogram: is it clean or smudgy? Smudgy spectrograms are sounds that have reflected off of various objects before making it to the microphone, and so smudgy spectrograms typically indicate songs from birds on the ground. A clean, sharp spectrogram is a good hint that the bird was in flight.
