import os
import numpy as np

import math
import os
from occupancy import get_rectangular_occupancy_map
from occupancy import get_circle_occupancy_map, log_circle_occupancy_map


class DataProcesser:

    def __init__(self,data_dir,observed_frame_num, predicting_frame_num): # ,dup_threshold):
 
        self.file_path = os.path.join(data_dir,'pixel_pos.csv')
        #self.dup_threshold = dup_threshold
        self.raw_data = None
        self.ped_num = None
        self.traj_data = []
        self.obs = []
        self.pred = []
        self.obs_length = observed_frame_num
        self.pred_length = predicting_frame_num

        self.from_csv()
        self.get_traj()
        self.get_obs_pred()


    def from_csv(self):
        print('Creating Raw Data from CSV file...')
        self.raw_data = np.genfromtxt(self.file_path, delimiter=',')
        # self.raw_data = np.genfromtxt(self.file_path).T
        self.ped_num = np.size(np.unique(self.raw_data[1, :]))

    def get_traj(self):
        """
        reshape data format from [frame_ID, ped_ID, y-coord, x-coord]
        to pedestrian_num * [ped_ID, frame_ID, x-coord, y-coord]
        """

        for pedIndex in range(self.ped_num):
            traj = []
            for i in range(len(self.raw_data[1])):
                if self.raw_data[1][i] == pedIndex + 1:
                    traj.append([self.raw_data[1][i], self.raw_data[0][i], self.raw_data[-1][i], self.raw_data[-2][i]])
            traj = np.reshape(traj, [-1, 4])

            #if self.traj_filter(traj, dup_threshold=self.dup_threshold):    don't know whats traj_filter or dup_threshold
            self.traj_data.append(traj)

        return self.traj_data

    def get_obs_pred(self):
        """
        get input observed data and output predicted data
        """
        count = 0

        for pedIndex in range(len(self.traj_data)):

            if len(self.traj_data[pedIndex]) >= self.obs_length + self.pred_length:
                obs_pedIndex = []
                pred_pedIndex = []
                count += 1
                for i in range(self.obs_length):
                    obs_pedIndex.append(self.traj_data[pedIndex][i])
                for j in range(self.pred_length):
                    pred_pedIndex.append(self.traj_data[pedIndex][j + self.obs_length])

                obs_pedIndex = np.reshape(obs_pedIndex, [self.obs_length, 4])
                pred_pedIndex = np.reshape(pred_pedIndex, [self.pred_length, 4])

                self.obs.append(obs_pedIndex)
                self.pred.append(pred_pedIndex)

        self.obs = np.reshape(self.obs, [count, self.obs_length, 4])
        self.pred = np.reshape(self.pred, [count, self.pred_length, 4])

        return self.obs, self.pred

    def get_traj_input(self):
        traj_input = []
        for pedIndex in range(len(self.obs)):
            person_pedIndex = []
            for i in range(self.obs_length):
                person_pedIndex.append([self.obs[pedIndex][i][-2], self.obs[pedIndex][i][-1]])
            person_pedIndex = np.reshape(person_pedIndex, [self.obs_length, 2])

            traj_input.append(person_pedIndex)

        traj_input = np.reshape(traj_input, [len(self.obs), self.obs_length, 2])

        return traj_input

    def get_expected_output(self):
        expected_ouput = []
        for pedIndex in range(len(self.pred)):
            person_pedIndex = []
            for i in range(self.pred_length):
                person_pedIndex.append([self.pred[pedIndex][i][-2], self.pred[pedIndex][i][-1]])
            person_pedIndex = np.reshape(person_pedIndex, [self.pred_length, 2])

            expected_ouput.append(person_pedIndex)

        expected_ouput = np.reshape(expected_ouput, [len(self.pred), self.pred_length, 2])

        return expected_ouput
    
    def save_files(self, data_dir): #data_dir here is combination of data_dir and dataset name
        np.save(data_dir +"/obs.npy", self.obs)
        np.save(data_dir +"/pred.npy", self.pred)


# NYGC processing
def file2matrix(filename):
    data = np.loadtxt(filename, dtype=int)
    data = np.reshape(data, [-1, 3])
    return data


def get_coord_from_txt(filename, ped_ID):
    data = file2matrix(filename)
    coord = []
    for i in range(len(data)):
        coord.append([ped_ID, data[i][-1], data[i][0], data[i][1]])
    coord = np.reshape(coord, [-1, 4])
    return coord


def select_trajectory(data, frame_num):
    if len(data) >= frame_num:
        return True
    else:
        return False


def get_all_trajectory(total_pedestrian_num):
    data = []

    for i in range(total_pedestrian_num):
        filename = str(i + 1).zfill(6) + '.txt'
        filepath = './data/NYGC/annotation/' + filename
        ped_ID = i + 1
        data.append(get_coord_from_txt(filepath, ped_ID))

    return data


def preprocess(data_dir):
    
    data = np.genfromtxt(data_dir, delimiter=',')
    numPeds = np.size(np.unique(data[1, :]))

    return data, numPeds


def get_traj_like(data, numPeds):
    '''
    reshape data format from [frame_ID, ped_ID, y-coord, x-coord]
    to pedestrian_num * [ped_ID, frame_ID, x-coord, y-coord]
    '''
    traj_data = []

    for pedIndex in range(numPeds):
        traj = []
        for i in range(len(data[1])):
            if data[1][i] == pedIndex + 1:
                traj.append([data[1][i], data[0][i], data[-1][i], data[-2][i]])
        traj = np.reshape(traj, [-1, 4])

        traj_data.append(traj)

    return traj_data


def get_traj_like_pixel(data, numPeds, dimension):
    '''
    reshape data format from [frame_ID, ped_ID, y-coord, x-coord]
    to pedestrian_num * [ped_ID, frame_ID, x-coord, y-coord]
    '''
    traj_data = []
    a = dimension[0]
    b = dimension[1]

    for pedIndex in range(numPeds):
        traj = []
        for i in range(len(data[1])):
            if data[1][i] == pedIndex + 1:
                traj.append([data[1][i], data[0][i], data[-1][i] * a, data[-2][i] * b])
        traj = np.reshape(traj, [-1, 4])

        traj_data.append(traj)

    return traj_data


def get_obs_pred_like(data, observed_frame_num, predicting_frame_num):
    """
    get input observed data and output predicted data
    """

    obs = []
    pred = []
    count = 0

    for pedIndex in range(len(data)):

        if len(data[pedIndex]) >= observed_frame_num + predicting_frame_num:
            obs_pedIndex = []
            pred_pedIndex = []
            count += 1
            for i in range(observed_frame_num):
                obs_pedIndex.append(data[pedIndex][i])
            for j in range(predicting_frame_num):
                pred_pedIndex.append(data[pedIndex][j + observed_frame_num])

            obs_pedIndex = np.reshape(obs_pedIndex, [observed_frame_num, 4])
            pred_pedIndex = np.reshape(pred_pedIndex, [predicting_frame_num, 4])

            obs.append(obs_pedIndex)
            pred.append(pred_pedIndex)

    obs = np.reshape(obs, [count, observed_frame_num, 4])
    pred = np.reshape(pred, [count, predicting_frame_num, 4])

    return obs, pred


def person_model_input(obs, observed_frame_num):
    person_model_input = []
    for pedIndex in range(len(obs)):
        person_pedIndex = []
        for i in range(observed_frame_num):
            person_pedIndex.append([obs[pedIndex][i][-2], obs[pedIndex][i][-1]])
        person_pedIndex = np.reshape(person_pedIndex, [observed_frame_num, 2])

        person_model_input.append(person_pedIndex)

    person_model_input = np.reshape(person_model_input, [len(obs), observed_frame_num, 2])

    return person_model_input


def model_expected_ouput(pred, predicting_frame_num):
    model_expected_ouput = []
    for pedIndex in range(len(pred)):
        person_pedIndex = []
        for i in range(predicting_frame_num):
            person_pedIndex.append([pred[pedIndex][i][-2], pred[pedIndex][i][-1]])
        person_pedIndex = np.reshape(person_pedIndex, [predicting_frame_num, 2])

        model_expected_ouput.append(person_pedIndex)

    model_expected_ouput = np.reshape(model_expected_ouput, [len(pred), predicting_frame_num, 2])

    return model_expected_ouput


def group_model_input(obs, observed_frame_num, neighborhood_size, dimensions, grid_size, raw_data):
    group_model_input = []

    for pedIndex in range(len(obs)):
        group_pedIndex = []
        for i in range(observed_frame_num):
            o_map_pedIndex = get_rectangular_occupancy_map(obs[pedIndex][i][1], obs[pedIndex][i][0], dimensions,  neighborhood_size, grid_size, raw_data)
            o_map_pedIndex = np.reshape(o_map_pedIndex, [int(neighborhood_size / grid_size) ** 2, ])
            group_pedIndex.append(o_map_pedIndex)
        group_pedIndex = np.reshape(group_pedIndex, [observed_frame_num, int(neighborhood_size / grid_size) ** 2])

        group_model_input.append(group_pedIndex)

    group_model_input = np.reshape(group_model_input, [-1, observed_frame_num, int(neighborhood_size / grid_size) ** 2])

    return group_model_input


def circle_group_model_input(obs, observed_frame_num, neighborhood_size, dimensions, neighborhood_radius, grid_radius, grid_angle, circle_map_weights, raw_data):
    group_model_input = []

    for pedIndex in range(len(obs)):
        group_pedIndex = []
        for i in range(observed_frame_num):
            o_map_pedIndex = get_circle_occupancy_map(obs[pedIndex][i][1], obs[pedIndex][i][0], dimensions, neighborhood_radius, grid_radius, grid_angle, raw_data)
            o_map_pedIndex = np.reshape(o_map_pedIndex, [-1, ])
            group_pedIndex.append(o_map_pedIndex)
        group_pedIndex = np.reshape(group_pedIndex, [observed_frame_num, -1])

        group_model_input.append(group_pedIndex)

    group_model_input = np.reshape(group_model_input, [len(group_model_input), observed_frame_num, -1])

    return group_model_input


def log_group_model_input(obs, observed_frame_num, neighborhood_size, dimensions, neighborhood_radius, grid_radius,
                          grid_angle, circle_map_weights, raw_data):
    group_model_input = []

    for pedIndex in range(len(obs)):
        group_pedIndex = []
        for i in range(observed_frame_num):
            o_map_pedIndex = log_circle_occupancy_map(obs[pedIndex][i][1], obs[pedIndex][i][0], dimensions,neighborhood_radius, grid_radius, grid_angle, raw_data)
            o_map_pedIndex = np.reshape(o_map_pedIndex, [-1, ])
            group_pedIndex.append(o_map_pedIndex)
        group_pedIndex = np.reshape(group_pedIndex, [observed_frame_num, -1])

        group_model_input.append(group_pedIndex)

    group_model_input = np.reshape(group_model_input, [len(group_model_input), observed_frame_num, -1])

    return group_model_input