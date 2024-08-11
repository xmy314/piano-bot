import pickle
from mido import MidiFile
import cv2
import numpy as np
from collections import namedtuple
import random
import os.path

# Declaring namedtuples
NoteEvent = namedtuple('NoteEvent', ['start_time', 'end_time', 'note'])
HandStateBrief = namedtuple('HandStateBrief', ['start_time', 'key'])
HandState = namedtuple('HandState', ['start_time', 'end_time', 'key'])

random.seed(4)

# 4*96 tick = 1 beat ~= 0.5s
# 4*1 beat = 1 bar
TICK_PER_INCREMENT = 96
INCREMENT_PER_BEAT = 4
BEAT_PER_SECOND = 2

LR_GAP = 4
MUTATION_PER_ITERATION = 10


TICK_PER_SECOND = TICK_PER_INCREMENT*INCREMENT_PER_BEAT*BEAT_PER_SECOND


def parse_events(track) -> list[NoteEvent]:
    events = []

    start = [-1]*88
    tick = 0
    for msg in track:
        tick += msg.time

        if msg.type == 'note_on':
            # offset by 21 such that the top is a0
            start[msg.note-21] = tick
        elif msg.type == "note_off":
            events.append(NoteEvent(start[msg.note-21], tick, msg.note-21))
            start[msg.note-21] = -1
        else:
            print(f"other midi event : {msg}")
            pass

    # (note, start time, end time)
    return events


def draw_track(note_events: list[NoteEvent], top_is_low: bool = True):
    total_tick = note_events[-1].end_time

    graphics = np.zeros((88, int(total_tick/TICK_PER_INCREMENT+10), 3))  # for visualization
    for note_event in note_events:
        # put red as default and override the played parts with green
        if top_is_low:
            graphics[note_event.note, note_event.start_time//TICK_PER_INCREMENT:note_event.end_time//TICK_PER_INCREMENT, :] = 150
        else:
            graphics[88-note_event.note, note_event.start_time//TICK_PER_INCREMENT:note_event.end_time//TICK_PER_INCREMENT, :] = 150
    return graphics


def key_to_note_index(key_index):
    lut = [0, 2, 4, 5, 7, 9, 11]
    return lut[(key_index-2) % 7]+((key_index-2)//7)*12+3


def note_to_servo_index_with_key(key_index, note_index):
    lut = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 100, 101]
    servo_index_from_hand_position = key_index*2
    servo_index_needed = lut[note_index]
    return servo_index_needed - servo_index_from_hand_position


def translatino_tick(start_key_index, end_key_index):
    # 0.6 seconds regard less of distance plus half a second per octave.
    # comletely guessed numbers.
    if start_key_index == end_key_index:
        return 0
    seconds_needed = 0.6+0.5/7*abs(start_key_index - end_key_index)
    return round(TICK_PER_SECOND*seconds_needed)


def hand_note_range(key_index):
    # low note is inclusive, high note is exclusive.
    low_note = key_to_note_index(key_index)
    high_note = key_to_note_index(key_index+8)

    return (low_note, high_note)


def get_position_index(hand_positions: list[HandState], time: int) -> int:
    position_index = -1
    for i in range(len(hand_positions)):
        if hand_positions[i][0] <= time and (i == len(hand_positions)-1 or time < hand_positions[i+1][0]):
            position_index = i
            break
    return position_index


def enrich_hands_positions(hands_positions_raw: list[list[HandStateBrief]]) -> list[list[HandState]]:
    hands_positions = []
    for hand_index in range(2):
        hand_positions_raw = hands_positions_raw[hand_index]
        hand_positions = []
        start_time = 0
        for i in range(len(hand_positions_raw)):

            if i != len(hand_positions_raw)-1:
                # if there is a next one
                if hand_positions_raw[i][1] == hand_positions_raw[i+1][1]:
                    # if the next is the same
                    # don't touch any thing
                    pass
                else:
                    # if the next is different
                    # calculate the time needed to next one.
                    tick_offset = translatino_tick(hand_positions_raw[i][1], hand_positions_raw[i+1][1])
                    # and end the current one
                    hand_position = HandState(start_time, hand_positions_raw[i+1][0]-tick_offset, hand_positions_raw[i][1])
                    hand_positions.append(hand_position)
                    # and set up the next one.
                    start_time = hand_positions_raw[i+1][0]
            else:
                #  if there is no next one, end the current one
                hand_position = HandState(start_time, 2**31-1, hand_positions_raw[i][1])
                hand_positions.append(hand_position)

        hands_positions.append(hand_positions)

    # (start time, end time, left most key index)
    return hands_positions


# The core of optimization, biased randomness
def mutate(hands_positions_raw: list[list[HandStateBrief]], last_note_start: int):

    # make a deep copy
    new_hands_positions_raw = [list(hands_positions_raw[0]), list(hands_positions_raw[1])]

    for i in range(random.randint(1, MUTATION_PER_ITERATION)):
        # select a hand at random
        hand_index = random.randint(0, 1)
        # select a mutation at random
        selection = random.random()
        if selection < 0.2:
            # chance to add something new
            # random note, random
            new_hands_positions_raw[hand_index].append(HandStateBrief(random.randint(1, last_note_start), random.randint(0, 51)))
            new_hands_positions_raw[hand_index].sort(key=lambda x: x[0])
        elif selection < 0.3:
            # chance to remove something old
            if len(new_hands_positions_raw[hand_index]) > 1:
                new_hands_positions_raw[hand_index].pop(random.randint(1, len(new_hands_positions_raw[hand_index])-1))
        elif selection < 0.95:
            # chance to make small adjustment
            position_index = random.randint(0, len(new_hands_positions_raw[hand_index])-1)
            secondary_selection = random.random()
            if secondary_selection > 0.75:
                note_shift = 1
            elif secondary_selection < 0.25:
                note_shift = -1
            else:
                note_shift = 0
            new_timing = 0 if position_index == 0 else max(new_hands_positions_raw[hand_index][position_index][0]+round((2*random.random()-1)*2*TICK_PER_INCREMENT), 1)

            new_hands_positions_raw[hand_index][position_index] = HandStateBrief(new_timing, new_hands_positions_raw[hand_index][position_index][1]+note_shift)
            new_hands_positions_raw[hand_index].sort(key=lambda x: x[0])
        else:
            # chance to move something.
            position_index = random.randint(0, len(new_hands_positions_raw[hand_index])-1)
            left_most_time = 0 if position_index == 0 else new_hands_positions_raw[hand_index][position_index-1][0]
            right_most_time = 0 if position_index == 0 else (last_note_start if position_index == len(new_hands_positions_raw[hand_index])-1 else new_hands_positions_raw[hand_index][position_index+1][0])

            new_hands_positions_raw[hand_index][position_index] = HandStateBrief(max(random.randint(left_most_time, right_most_time), 0), random.randint(0, 51-7))

    # force the first one to start at 0.
    new_hands_positions_raw[0][0] = HandStateBrief(0, new_hands_positions_raw[0][0][1])
    new_hands_positions_raw[1][0] = HandStateBrief(0, new_hands_positions_raw[1][0][1])

    # sort by time.
    for i in range(2):
        new_hands_positions_raw[i].sort(key=lambda x: x[0])

    return new_hands_positions_raw


def check_requirements(hands_positions_raw):
    # for each hand, the period in between movement should be large enough for a few key presses.
    hands_positions = enrich_hands_positions(hands_positions_raw)

    for hand_index in range(2):
        for i in range(len(hands_positions[hand_index])):
            if i == len(hands_positions[hand_index])-1 or hands_positions[hand_index][i][1] < hands_positions[hand_index][i+1][0]:
                pass
            else:
                return False

    left_index = 0
    left_main = True
    right_index = 0
    right_main = True
    while True:
        # check collision
        left_hand_right_end = hands_positions[0][left_index][2]+7+LR_GAP if left_main else max(hands_positions[0][left_index][2], hands_positions[0][left_index+1][2])+7+LR_GAP
        right_hand_left_end = hands_positions[1][right_index][2] if right_main else min(hands_positions[1][right_index][2], hands_positions[1][right_index+1][2])
        if left_hand_right_end >= right_hand_left_end:
            return False

        if left_index == len(hands_positions[0])-1 and right_index == len(hands_positions[1])-1:
            break

        left_time = hands_positions[0][left_index][1] if left_main else hands_positions[0][left_index+1][0]
        right_time = hands_positions[1][right_index][1] if right_main else hands_positions[1][right_index+1][0]
        if left_time < right_time:
            if not left_main:
                left_index += 1
            left_main = not left_main
        else:
            if not right_main:
                right_index += 1
            right_main = not right_main

    return True


def eval_position(hands_positions: list[list[HandState]], note_events: list[NoteEvent]):
    # ideally
    # criterias
    # hit the correct note in the correct octave
    # hit the correct note in neightbouring octave
    # hit all notes
    # hit the note for as long as the midi says.

    # loop through events
    # check at the beginning if it is covered.
    # check if hand moved before note is done.
    score: float = 0
    for note_event in note_events:
        for hand_index in range(2):
            # get the index that would be responsible for the time that the hand is there.
            position_index = get_position_index(hands_positions[hand_index], note_event.start_time)
            note_index_up, note_index_bottom = hand_note_range(hands_positions[hand_index][position_index][2])

            if note_event.note < note_index_up or note_index_bottom <= note_event.note:
                # if hand doesn't cover it, it is no use
                continue

            if note_event.start_time > hands_positions[hand_index][position_index][1]:
                # if hand is moving to next position, it is no use
                continue

            # it can press the key down, award it
            score += 0.2

            # if hand move before note is complete, award accordingly, else award it.
            if note_event.end_time > hands_positions[hand_index][position_index][1]:
                ratio = (hands_positions[hand_index][position_index][1]-note_event.start_time)/(note_event.end_time-note_event.start_time)
                score += 0.8*(2*ratio - ratio**2)
            else:
                score += 0.8*(1)

    return score


def draw_track_play(hands_positions, note_events: list[NoteEvent], top_is_low: bool = True):
    total_tick = note_events[-1].end_time

    graphics = np.zeros((88, int(total_tick/TICK_PER_INCREMENT+10), 3))  # for visualization

    for hand_positions in hands_positions:
        for hand_position in hand_positions:
            start_time, end_time, key_index = hand_position
            start_col = start_time//TICK_PER_INCREMENT
            end_col = min(end_time//TICK_PER_INCREMENT, graphics.shape[1])

            occupied_index_low = max(key_to_note_index(key_index-2), 0)  # inclusive
            playable_index_low = max(key_to_note_index(key_index), 0)  # inclusive
            playable_index_high = min(key_to_note_index(key_index+8), 87)  # exclusive
            occupied_index_high = min(key_to_note_index(key_index+10), 87)  # exclusive

            if top_is_low:
                graphics[occupied_index_low:occupied_index_high, start_col:end_col, :] = [20, 20, 40]
                graphics[playable_index_low:playable_index_high, start_col:end_col, :] = 50
            else:
                graphics[89-occupied_index_high+1:89-occupied_index_low, start_col:end_col, :] = [20, 20, 40]
                graphics[89-playable_index_high+1:89-playable_index_low, start_col:end_col, :] = 50

    for note_event in note_events:
        # put red as default and override the played parts with green
        if top_is_low:
            graphics[note_event.note, note_event.start_time//TICK_PER_INCREMENT:note_event.end_time//TICK_PER_INCREMENT, :] = [0, 0, 150]
        else:
            graphics[88-note_event.note, note_event.start_time//TICK_PER_INCREMENT:note_event.end_time//TICK_PER_INCREMENT, :] = [0, 0, 150]

        for hand_index in range(2):
            # get the index that would be responsible for the time that the hand is there.
            position_index = get_position_index(hands_positions[hand_index], note_event.start_time)
            note_index_up, note_index_bottom = hand_note_range(hands_positions[hand_index][position_index][2])

            if note_event.note < note_index_up or note_index_bottom <= note_event.note:
                # if hand doesn't cover it, it is no use
                continue

            if note_event.start_time > hands_positions[hand_index][position_index][1]:
                # if hand is moving to next position, it is no use
                continue

            start_col = note_event.start_time//TICK_PER_INCREMENT
            end_col = min(note_event.end_time, hands_positions[hand_index][position_index][1])//TICK_PER_INCREMENT

            if top_is_low:
                graphics[note_event.note, start_col:end_col, :] = [0, 140, 0]
            else:
                graphics[88-note_event.note, start_col:end_col, :] = [0, 140, 0]

    return graphics


def convert_to_arduino(hands_positions, note_events: list[NoteEvent]):
    template = {
        "left_hand_motor_to": 0,
        "right_hand_motor_to": 1,
        "left_hand_motor_off": 2,
        "right_hand_motor_off": 3,
        "left_hand_servo_up": 4,
        "right_hand_servo_up": 5,
        "left_hand_servo_down": 6,
        "right_hand_servo_down": 7,
    }

    # (time, code)
    # stepper motor need to be fliped
    command_list = [

    ]

    for note_event in note_events:
        for hand_index in range(2):
            # get the index that would be responsible for the time that the hand is there.
            position_index = get_position_index(hands_positions[hand_index], note_event.start_time)
            note_index_up, note_index_bottom = hand_note_range(hands_positions[hand_index][position_index][2])

            if note_event.note < note_index_up or note_index_bottom <= note_event.note:
                # if hand doesn't cover it, it is no use
                continue

            if note_event.start_time > hands_positions[hand_index][position_index][1]:
                # if hand is moving to next position, it is no use
                continue

            servo_index = note_to_servo_index_with_key(hands_positions[hand_index][position_index][2], note_event.note)

            if hand_index == 0:
                command_list.append((note_event.start_time, template["left_hand_servo_down"], servo_index))
            else:
                command_list.append((note_event.start_time, template["right_hand_servo_down"], servo_index))

            effective_end_time = min(note_event.end_time, hands_positions[hand_index][position_index][1])

            if hand_index == 0:
                command_list.append((effective_end_time, template["left_hand_servo_up"], servo_index))
            else:
                command_list.append((effective_end_time, template["right_hand_servo_up"], servo_index))

    # move to location
    command_list.append((-TICK_PER_SECOND*2, template["left_hand_motor_to"], round(2.3*1000/61.5*hands_positions[0][0].key)))
    command_list.append((-TICK_PER_SECOND*2, template["right_hand_motor_to"], round(2.3*1000/61.5*hands_positions[1][0].key)))
    command_list.append((0, template["left_hand_motor_off"], 0))
    command_list.append((0, template["right_hand_motor_off"], 0))

    # left hand
    for position_index in range(len(hands_positions[0])-1):
        # for all but the last state
        # add command to move to next state
        command_list.append((hands_positions[0][position_index].end_time, template["left_hand_motor_to"], round(2.3*1000/61.5*hands_positions[0][position_index+1].key)))
        # add command to shut off stepper motor after the movement
        command_list.append((hands_positions[0][position_index+1].start_time, template["left_hand_motor_off"], 0))

    # right hand
    for position_index in range(len(hands_positions[1])-1):
        # all but the first movement
        command_list.append((hands_positions[1][position_index].end_time, template["right_hand_motor_to"], round(2.3*1000/61.5*hands_positions[1][position_index+1].key)))
        # add command to shut off stepper motor after the movement
        command_list.append((hands_positions[1][position_index+1].start_time, template["left_hand_motor_off"], 0))

    command_list.sort(key=lambda x: x[0])

    transposed = []
    transposed.append([int((command_list[i][0]-command_list[0][0])*1000/TICK_PER_SECOND) for i in range(len(command_list))])
    transposed.append([command_list[i][1] for i in range(len(command_list))])
    transposed.append([command_list[i][2] for i in range(len(command_list))])

    str0 = str(transposed[0]).replace("[", "").replace("]", "")
    str1 = str(transposed[1]).replace("[", "").replace("]", "")
    str2 = str(transposed[2]).replace("[", "").replace("]", "")
    ret = f"const PROGMEM int song_inst_count = {len(transposed[0])};\n"
    ret += f"const PROGMEM long song_time[] ={{ {str0} }};\n"
    ret += f"const PROGMEM int song_sync[] ={{ {str1} }};\n"
    ret += f"const PROGMEM int song_args[] ={{ {str2} }};\n"

    return ret


# get the piano track and convert it to a list of events.
mid = MidiFile('midis/Beethoven - Moonlight Sonata (1st Movement) .mid')  # https://onlinesequencer.net/4106657
print(f"mid file : {mid}")
for track in mid.tracks:
    print(f"mid track : {track}")
selected_track = mid.tracks[0]  # SONG_SPECIFIC: pick the track that is the piano. It is either in the name or has the largest number of messages.
events = parse_events(selected_track)
# visualize the track
graphics = draw_track(events)
cv2.imwrite(os.path.join("output", "visual.png"), graphics)


# get the previous best play plan if it exists, else get some random thing.
if os.path.isfile('start_point.pickle'):
    with open('start_point.pickle', 'rb') as handle:
        best_placement: list[list[HandStateBrief]] = pickle.load(handle)
else:
    # start with something essentially random
    best_placement: list[list[HandStateBrief]] = mutate([
        [HandStateBrief(0, 0)],  # left hand on a0
        [HandStateBrief(0, 2+3*7)]  # right hand on c4
    ], events[-1][1])

hands_positions: list[list[HandState]] = enrich_hands_positions(best_placement)
highest_score = eval_position(hands_positions, events)

c_snip = convert_to_arduino(hands_positions, events)
with open(os.path.join("output", "snip.c"), "w+") as fi:
    fi.write(c_snip)

graphics = draw_track_play(hands_positions, events)
cv2.imwrite(os.path.join("output", "played.png"), graphics)

with open(os.path.join("output", "record.pickle"), 'wb') as handle:
    pickle.dump(best_placement, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"starting at :{highest_score:7.3f}, {highest_score/len(events):6.2f}")

while True:
    while True:
        placement = mutate(best_placement, events[-1][1])
        if check_requirements(placement):
            break

    hands_positions = enrich_hands_positions(placement)
    score = eval_position(hands_positions, events)

    if score > highest_score:
        print(f"improved to :{highest_score:7.3f}, {highest_score/len(events):6.2f}")

        highest_score = score
        best_placement = placement

        graphics = draw_track_play(hands_positions, events)
        cv2.imwrite(os.path.join("output", "played.png"), graphics)

        with open(os.path.join("output", "current_best.pickle"), 'wb') as handle:
            pickle.dump(best_placement, handle, protocol=pickle.HIGHEST_PROTOCOL)
