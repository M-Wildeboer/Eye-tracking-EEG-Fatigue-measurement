import pandas as pd
import numpy as np
import cv2
import sys
from colour import Color

# Load the eye-tracking data, click data, and AOI definitions
data_raw = pd.read_csv('data/data.csv')
clicks = pd.read_csv('data/clicks.csv')
aoi = pd.read_csv('data/aoi.csv')

# Data heads are: Exp,Part,trial_index,Stim,time,xL,yL,xL_pro,yL_pro,xR,yR,xR_pro,yR_pro,xF,yF,xF_pro,yF_pro,inside_stimulus
# data heads for aoi: Stimulus,AOI,x,y,w,h
# clicks heads: Exp,Part,trial_index,Stim,stim_start_time,click_time,reaction_time,click_x_stim,click_y_stim

# store clicks as dict with key as stimuli and value as x y of click
click_pos_dict = {}
for _, row in clicks.iterrows():
    stim = row['Stim']
    click_x, click_y = row['click_x_stim'], row['click_y_stim']
    if stim not in click_pos_dict:
        click_pos_dict[stim] = (click_x, click_y)

# make dict for AOI with key as stim and value new dict with aoi as key and x,y,w,h as value
aoi_dict = {}
for _, row in aoi.iterrows():
    stim = row['Stimulus']
    aoi_name = row['AOI']
    x, y, w, h = row['x'], row['y'], row['w'], row['h']
    if stim not in aoi_dict:
        aoi_dict[stim] = {}
    aoi_dict[stim][aoi_name] = (x, y, w, h)

# Split the raw dataset into separate trial-level datasets
data_trials = []
for exp in data_raw['Exp'].unique():
    for part in data_raw['Part'].unique():
        for trial_random in data_raw['trial_index'].unique():
            trial_data = data_raw[(data_raw['Exp'] == exp) & (data_raw['Part'] == part) & (data_raw['trial_index'] == trial_random)]
            data_trials.append(trial_data)

def point_in_rect(x, y, rect):
    rect_x, rect_y, rect_w, rect_h = rect
    return rect_x <= x < rect_x + rect_w and rect_y <= y < rect_y + rect_h

# sort data trials by stimulus name
data_trials.sort(key=lambda x: x['Stim'].iloc[0])
# Stimuli is in data/stimuli/waldo_001.png etc. Show the stimulus from INDEX
# RAND_INDEX = 3
# img_path = "data/stimuli/" + random_trials[RAND_INDEX]['Stim'].iloc[0]

# Prepare lists to store the output summaries for each trial
insight_data = []
paths_data = []
path_insight_data = []
accuracy = []
for i, trial in enumerate(data_trials):
    # visited = [('aoi_name', fix_count)]
    visited = []
    visit_index = 0
    path_insight = {} # ({aoi_name: {'num_fixations': int, 'num_revisits': int}})

    index = i
    img_path = "data/stimuli/" + data_trials[index]['Stim'].iloc[0]

    # initialize path_insight for this trial
    for aoi_name in aoi_dict.get(data_trials[index]['Stim'].iloc[0], {}):
        path_insight[aoi_name] = {'num_fixations': 0, 'num_revisits': 0}

    RADIUS = 5
    THICKNESS = -1
    DARKNESS = 0.25
    color = (0, 255, 0)

    # Load the stimulus image and darken it so fixations and AOIs are easier to see
    img = cv2.imread(img_path)
    img = cv2.convertScaleAbs(img, alpha=DARKNESS, beta=0)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    img_with_circles = img.copy()

    trial_data = data_trials[index]

    data_oob = 0

    # Create a color gradient across fixations to show the time order of the scanpath
    purple = Color("red")
    colors = list(purple.range_to(Color("yellow"),len(trial_data)))

    prev_aoi = None

    # Draw all fixations for data trial
    for i, (_, row) in enumerate(trial_data.iterrows()):
        xF, yF = int(row['xF']), int(row['yF'])

        color = (colors[i].blue * 255, colors[i].green * 255, colors[i].red * 255)

        x = xF
        y = yF

        cur_aoi = "other" if point_in_rect(x, y, (0, 0, w, h)) else "oob"
        for aoi_name, (aoi_x, aoi_y, aoi_w, aoi_h) in aoi_dict[trial_data['Stim'].iloc[0]].items():
            if point_in_rect(x, y, (aoi_x, aoi_y, aoi_w, aoi_h)):
                cur_aoi = aoi_name
                break

        if cur_aoi in path_insight:
            path_insight[cur_aoi]['num_fixations'] += 1
            if cur_aoi != prev_aoi and path_insight[cur_aoi]['num_fixations'] > 1:
                path_insight[cur_aoi]['num_revisits'] += 1
        
        if i == 0:
            visited.append((cur_aoi, 1))
        else:
            if cur_aoi == prev_aoi:
                visited[visit_index] = (cur_aoi, visited[visit_index][1] + 1)
            else:
                visit_index += 1
                visited.append((cur_aoi, 1))



        prev_aoi = cur_aoi        

        if cur_aoi == "oob":
            data_oob += 1
            continue
        cv2.circle(img_with_circles, (x, y), RADIUS, color, THICKNESS)
        if i != 0:
            prev_xF, prev_yF = int(trial_data.iloc[i-1]['xF']), int(trial_data.iloc[i-1]['yF'])
            cv2.line(img_with_circles, (prev_xF, prev_yF), (xF, yF), color, 2)
            

    # Draw all AOI boxes on top of the stimulus image
    # Waldo is green, salient AOIs are yellow, and striped AOIs are pink
    for aoi_name, (x, y, w, h) in aoi_dict[trial_data['Stim'].iloc[0]].items():
        color = (100, 255, 100) if aoi_name == 'Waldo' else (255, 255, 20) if aoi_name.startswith("Sa") else (255, 100, 165)
        cv2.rectangle(img_with_circles, (x, y), (x + w, y + h), color, 2)
    
    # Draw the participant's click position and check whether it fell inside the Waldo AOI
    clicked_in_waldo = -1
    click_pos = click_pos_dict.get(trial_data['Stim'].iloc[0], (0, 0))
    click_pos = (int(click_pos[0]), int(click_pos[1]))
    if (click_pos != (0, 0)):
        cv2.circle(img_with_circles, click_pos, RADIUS * 2, (255, 0, 0), THICKNESS)  # Blue for click position
        clicked_in_waldo = 1 if point_in_rect(click_pos[0], click_pos[1], aoi_dict[trial_data['Stim'].iloc[0]]['Waldo']) else 0
    
    accuracy.append({
        'stimulus': trial_data['Stim'].iloc[0],
        'clicked_in_waldo': clicked_in_waldo
    })
    
    print(f"Out of bounds fixations - Data: {data_oob} - Total fixations: {len(trial_data)}")
    
    # Store summary values for later analysis
    insight_data.append({
        'stimulus': trial_data['Stim'].iloc[0],
        'data_oob': data_oob,
        'total_fixations': len(trial_data),
        'rel-missed': data_oob / len(trial_data) if len(trial_data) > 0 else 0 
    })
    paths_data.append({
        'stimulus': trial_data['Stim'].iloc[0],
        'visited': visited
    })

    path_insight_data.append({
        'stimulus': trial_data['Stim'].iloc[0],
        'path_insight': path_insight
    })

    # Store image in data/processed_images/ with same name as stimulus but with _processed suffix
    output_path = "./"+img_path.replace("stimuli", "processed_images").replace(".png", "_processed.png")
    cv2.imwrite(output_path, img_with_circles)


# store insight data in insights.csv
insight_df = pd.DataFrame(insight_data)
insight_df.to_csv('insights.csv', index=False)

paths_df = pd.DataFrame(paths_data)
paths_df.to_csv('paths.csv', index=False)

path_insight_df = pd.DataFrame(path_insight_data)
path_insight_df.to_csv('path_insights.csv', index=False)

accuracy_df = pd.DataFrame(accuracy)
accuracy_df.to_csv('accuracy.csv', index=False)



