# Test sentences for background-noise recording

Read aloud with `python scripts/record_samples.py --sentences`. Every top-level
`- ` line in this file is one prompt — the script reads them in order, records
a 3-second clip per sentence, and slices each into five 1-second training
clips for `_background_noise` (see `slice_windows()` in
[record_samples.py](../scripts/record_samples.py)).

**Never say the actual keywords** (`music`, `play`, `pause`, `skip`) while
reading these. The entire point of this class is speech that is *not* a
keyword — saying one here would put a mislabeled positive example into the
rejection class.

Read at a normal conversational pace and volume, the way you'd actually talk
near the device — not word-by-word.

## Everyday sentences

Ordinary conversation, unrelated to the device.

- I think it's going to rain again this afternoon.
- Can you grab the mail on your way in?
- We should really clean out the garage this weekend.
- I have no idea where I put my keys.
- The meeting got pushed back to three o'clock.
- Traffic was terrible on the way home today.
- Let's order pizza instead of cooking tonight.
- Did you see the game last night?
- I'm going to need another cup of coffee.
- The dog wants to go outside again.
- My phone battery is almost dead.
- It's colder in here than I expected.

## Words that sound like your keywords

Real sentences built around words that share sounds with `music`, `play`,
`pause`, and `skip` — the pairs most likely to fool the model if it never
hears them.

- Please pass the plate down to the end of the table.
- I need to pause and catch my breath for a second.
- Can you clear off that plate before dinner?
- The kids like to play in the yard after school.
- We had to pass on dessert, we were too full.
- She wore a bracelet made of polished glass.
- The bus made an unexpected stop near the museum.
- Just skip to the good part of the story.
- I slipped on the wet floor by the sink.
- The ship left the harbor early this morning.
- He took a sip of coffee before answering.
- That chip in the paint needs to be fixed.
- We watched the moose from a safe distance.
- The juice was a little too sour for me.

## Commands and questions

Ordinary voice-assistant-shaped phrases, none of which are your keywords.

- Turn on the lights in the kitchen.
- What's the weather going to be like tomorrow?
- Set a timer for five minutes.
- Remind me to call the dentist later.
- How far is it to the nearest gas station?
- Add milk and eggs to the shopping list.
- What time does the store close tonight?
- Could you turn the volume down a little?
