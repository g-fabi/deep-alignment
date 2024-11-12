import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

class OpportunityPlusPlusRaw():
    """
    Processes the Opportunity++ dataset and saves the data in the desired format.
    """

    def __init__(self, data_path, destination) -> None:
        self.data_path = os.path.join(data_path, 'data')
        self.destination = destination
        label_legend_path = os.path.join(data_path, 'label_legend.txt')
        self.label_dict, self.category_label_mappings = read_label_legend(label_legend_path)

        #zero based indexing
        self.label_columns = {
            'Locomotion': 243,       
            'HL_Activity': 244,      
            'LL_Left_Arm': 245,      
            'LL_Right_Arm': 247,     
            'ML_Both_Arms': 249      
        }
        # we dont care about '_object' categories
        self.categories_to_process = ['Locomotion', 'HL_Activity', 'LL_Left_Arm', 'LL_Right_Arm', 'ML_Both_Arms']

    def process_dataset(self):
        print('Processing inertial data...')
        self.process_inertial_data()
        print('Inertial data processing complete.')
        print('Processing pose data...')
        self.process_pose_data()
        print('Pose data processing complete.')
        print('Checking modalities alignment...')
        self.check_modalities_alignment()
        print('Modalities alignment check complete.')

    def process_inertial_data(self):
        sessions = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        sessions.sort()
        for session in tqdm(sessions, desc='Processing Inertial Data'):
            session_path = os.path.join(self.data_path, session)
            sensors_file = os.path.join(session_path, f'{session}_sensors_data.txt')
            sensors_df = pd.read_csv(sensors_file, sep='\s+', header=None, na_values='NaN')
            sensor_columns = list(range(1, 134))  # Body-worn sensors
            label_columns = [self.label_columns[cat] for cat in self.categories_to_process]
            columns_to_keep = sensor_columns + label_columns
            sensors_df = sensors_df.iloc[:, columns_to_keep]
            for category in self.categories_to_process:
                label_col = self.label_columns[category]
                label_mapping = self.category_label_mappings.get(category, {})
                sensors_df['label'] = sensors_df.iloc[:, len(sensor_columns) + self.categories_to_process.index(category)]
                sensors_df['label_shift'] = sensors_df['label'].shift(1)
                sensors_df['segment'] = (sensors_df['label'] != sensors_df['label_shift']).cumsum()
                grouped = sensors_df.groupby(['segment', 'label'])
                trial_counts = {}
                for (segment_id, label_value), group in grouped:
                    if label_value == 0 or label_value not in label_mapping:
                        continue
                    label_name = label_mapping[label_value]
                    trial_counts.setdefault(label_name, 0)
                    trial_counts[label_name] += 1
                    trial_num = trial_counts[label_name]
                    subject, session_name = session.split('-')
                    dest_folder = os.path.join(self.destination, subject, session_name, 'Inertial', category, label_name)
                    os.makedirs(dest_folder, exist_ok=True)
                    dest_filename = f'imu_{label_name}_{trial_num}.csv'
                    dest_path = os.path.join(dest_folder, dest_filename)
                    data_segment = group.iloc[:, :len(sensor_columns)]
                    data_segment.to_csv(dest_path, header=False, index=False)
                sensors_df.drop(['label', 'label_shift', 'segment'], axis=1, inplace=True)
                
    def process_pose_data(self):
        sessions = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        sessions.sort()
        self.empty_segments = []
        for session in tqdm(sessions, desc='Processing Pose Data'):
            session_path = os.path.join(self.data_path, session)
            pose_file = os.path.join(session_path, f'{session}_pose.csv')
            if not os.path.exists(pose_file):
                continue
            pose_df = pd.read_csv(pose_file)
            # filter rows where Participant == 1, since they are the subject
            pose_df = pose_df[pose_df['Participant'] == 1].copy()
            pose_df['Time'] = pose_df['Frame_Number'] * 100  # 10 fps, each frame is 100 ms
            srt_files = [f for f in os.listdir(session_path) if f.endswith('.srt')]
            srt_files = [f for f in srt_files if self.get_category_from_filename(f) in self.categories_to_process]
            trial_counts = {}
            for srt_file in srt_files:
                category = self.get_category_from_filename(srt_file)
                if category not in self.categories_to_process:
                    continue
                srt_file_path = os.path.join(session_path, srt_file)
                labels = parse_srt_file(srt_file_path)
                label_mapping = self.category_label_mappings.get(category, {})
                for label in labels:
                    label_id = label['label_id']
                    if label_id not in label_mapping:
                        continue
                    label_name = label_mapping[label_id]
                    trial_counts.setdefault((category, label_name), 0)
                    trial_counts[(category, label_name)] += 1
                    trial_num = trial_counts[(category, label_name)]
                    start_time_ms = label['start_time']
                    end_time_ms = label['end_time']
                    data_segment = pose_df[(pose_df['Time'] >= start_time_ms) & (pose_df['Time'] <= end_time_ms)]
                    if data_segment.empty:
                        self.empty_segments.append({
                            'session': session,
                            'category': category,
                            'label_name': label_name,
                            'label_id': label_id,
                            'start_time_ms': start_time_ms,
                            'end_time_ms': end_time_ms
                        })
                        continue
                    subject_session = session.split('-')
                    if len(subject_session) == 2:
                        subject, session_name = subject_session
                    else:
                        continue
                    dest_folder = os.path.join(self.destination, subject, session_name, 'Skeleton', category, label_name)
                    os.makedirs(dest_folder, exist_ok=True)
                    dest_filename = f'pose_{label_name}_{trial_num}.csv'
                    dest_path = os.path.join(dest_folder, dest_filename)
                    data_segment.to_csv(dest_path, header=True, index=False)
        # print(f"Total empty segments: {len(self.empty_segments)}")

    def check_modalities_alignment(self):
        for subject in os.listdir(self.destination):
            subject_path = os.path.join(self.destination, subject)
            if not os.path.isdir(subject_path):
                continue
            for session in os.listdir(subject_path):
                session_path = os.path.join(subject_path, session)
                if not os.path.isdir(session_path):
                    continue
                modalities = ['Inertial', 'Skeleton']
                modality_paths = {modality: os.path.join(session_path, modality) for modality in modalities}
                if not all(os.path.exists(modality_paths[modality]) for modality in modalities):
                    continue
                categories = self.categories_to_process
                for category in categories:
                    category_paths = {modality: os.path.join(modality_paths[modality], category) for modality in modalities}
                    if not all(os.path.exists(category_paths[modality]) for modality in modalities):
                        continue
                    label_names = set()
                    for modality in modalities:
                        labels_in_modality = os.listdir(category_paths[modality])
                        label_names.update(labels_in_modality)
                    for label_name in label_names:
                        label_paths = {modality: os.path.join(category_paths[modality], label_name) for modality in modalities}
                        if not all(os.path.exists(label_paths[modality]) for modality in modalities):
                            continue
                        num_files = {}
                        for modality in modalities:
                            files = [f for f in os.listdir(label_paths[modality]) if os.path.isfile(os.path.join(label_paths[modality], f))]
                            num_files[modality] = len(files)
                        if num_files['Inertial'] != num_files['Skeleton']:
                            print(f"Discrepancy in {subject}/{session}/{category}/{label_name}: Inertial has {num_files['Inertial']} files, Skeleton has {num_files['Skeleton']} files.")

    def get_category_from_filename(self, filename):
        filename = filename.lower()
        if 'locomotion' in filename:
            return 'Locomotion'
        elif 'hl_activity' in filename:
            return 'HL_Activity'
        elif 'left_arm.srt' in filename and 'object' not in filename:
            return 'LL_Left_Arm'
        elif 'right_arm.srt' in filename and 'object' not in filename:
            return 'LL_Right_Arm'
        elif 'ml_both_arms' in filename:
            return 'ML_Both_Arms'
        else:
            return None

def read_label_legend(label_legend_path):
    label_dict = {}
    category_label_mappings = {}
    with open(label_legend_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith(('Unique index', 'Data columns:', 'Label columns:')):
            continue
        parts = line.split('   -   ')
        if len(parts) != 3:
            continue
        label_id_str, category_str, label_name = parts
        category = category_str.strip()
        if '_object' in category:
            continue
        label_id = int(label_id_str.strip())
        label_name = label_name.strip()
        label_dict[label_id] = (category, label_name)
        category_label_mappings.setdefault(category, {})
        category_label_mappings[category][label_id] = label_name
    return label_dict, category_label_mappings

def parse_srt_file(srt_file_path):
    labels = []
    with open(srt_file_path, 'r') as f:
        lines = f.readlines()
    i = 0
    n = len(lines)
    while i < n:
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        index_line = lines[i].strip()
        i += 1
        if i >= n:
            break
        time_line = lines[i].strip()
        i += 1
        if i >= n:
            break
        label_line = lines[i].strip()
        i += 1
        while i < n and lines[i].strip():
            i += 1
        try:
            index = int(index_line)
        except ValueError:
            continue
        start_time_str, end_time_str = time_line.split(' --> ')
        start_time = time_str_to_milliseconds(start_time_str)
        end_time = time_str_to_milliseconds(end_time_str)
        label_parts = label_line.split(' - ')
        try:
            label_id = int(label_parts[0])
            label_name = ' - '.join(label_parts[1:])
        except ValueError:
            continue
        labels.append({'index': index, 'start_time': start_time, 'end_time': end_time, 'label_id': label_id, 'label_name': label_name})
    return labels

def time_str_to_milliseconds(time_str):
    hh, mm, ss_ms = time_str.split(':')
    ss, ms = ss_ms.split(',')
    total_milliseconds = ((int(hh) * 3600 + int(mm) * 60 + int(ss)) * 1000) + int(ms)
    return total_milliseconds

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='initial data path', required=True)
    parser.add_argument('--destination_path', type=str, help='destination path', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    opp = OpportunityPlusPlusRaw(args.data_path, args.destination_path)
    opp.process_dataset()
