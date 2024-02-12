import pickle
import sys
import os

from feature import Motion, HeartRate, Step
from label import Label
from data_handler import read_files
from subject import Subject, SubjectCollector
from train import Train


def read_data(use_prepared):
    """Reads data from path

    Parameters
    ----------
    use_prepared : bool
        True if we should try using the already prepared data (if exists)

    Returns
    -------
    tuple
        labels, motions, heart_rates, steps, subjects objects
    """
    labels_dict = read_files("labels")
    labels = []
    motions = []
    heart_rates = []
    steps = []
    subjects = []
    num_of_files = len(labels_dict)
    for idx, (key, value) in enumerate(labels_dict.items()):
        print('Loading file {}/{}'.format(idx + 1, num_of_files), end='\r')
        label = Label(key, value)
        labels.append(label)
        motions.append(
            Motion(
                key,
                read_files(
                    "motions",
                    idx=key,
                    use_prepared=use_prepared),
                label.start_time,
                label.end_time,
                label.prepared_data,
                prepared_used=use_prepared))
        steps.append(
            Step(
                key,
                read_files(
                    "steps",
                    idx=key,
                    use_prepared=use_prepared),
                label.start_time,
                label.end_time,
                label.prepared_data,
                prepared_used=use_prepared))
        steps[-1]
        heart_rates.append(
            HeartRate(
                key,
                read_files(
                    "heart_rates",
                    idx=key,
                    use_prepared=use_prepared),
                label.start_time,
                label.end_time,
                label.prepared_data,
                prepared_used=use_prepared))
        heart_rates[-1]
        s = Subject(labels[-1], motions[-1], heart_rates[-1], steps[-1])
        if not use_prepared:
            s.save_normalized_data()
        subjects.append(s)
    print()
    return labels, motions, heart_rates, steps, subjects

def get_train_vals(filenames):
    for fname in filenames:
        with open(fname, 'rb') as f:
            train = pickle.load(f)
            print('--------- ', fname, ' ---------')
            print(train.loss, train.accuracy)
            
# def predict(heart_rate, motion, step):
#     hr = HeartRate(1, [0, heart_rate], 0, 1, None)
#     m = Motion(1, [0, motion], 0, 1, None)
#     s = Step(1, [0, step], 0, 1, None)
    

def check_actual_vs_predicted(data):
    """Compares labeled data with prediction

    Parameters
    ----------
    data : pandas.DataFrame
        labeled data, use subject_collector.generate_sleep_data()
    """
    vls =[]
    oks = []
    for i in range(0, int(len(data["heart_rate"] * 0.14)), 15):
        heart_rate = data['heart_rate'][i]
        motion = data['motion'][i]
        step = data['step'][i]
        val = train.predict_sleep_phase(
            heart_rate,
            motion,
            step)
        if val == 4:
            val = 5
        act_val = data['sleep_phase'][i]
        print(
            act_val,
            "--- vs ---", val)
        oks.append(int(val == act_val))
        vls.append(val)
        
    temp_dict = dict.fromkeys([val for val in vls], 0)
    for x in temp_dict.keys():
        temp_dict[x] = str((100 * list(vls).count(x)) / len(vls)) + '%'  
          
    print('Found labels percentage:')
    print(temp_dict)
        
    temp_dict = dict.fromkeys([val for val in data['sleep_phase']], 0)
    for x in temp_dict.keys():
        temp_dict[x] = str((100 * list(data['sleep_phase']).count(x)) / len(data['sleep_phase'])) + '%'
    print('Actual labels percentage:')
    print(temp_dict)
    print('Found {} out of {}'.format(sum(oks), len(oks)))


if __name__ == "__main__":
    try:
        # Because preparing the data took a lot of time, I saved the prepared
        # data, so first of all I will try to load the already prepared data if
        # exists
        labels, motions, heart_rates, steps, subjects = read_data(use_prepared=True)
    except BaseException:
        # If it does not exist I will prepare the data here
        print("Prepared data not found")
        labels, motions, heart_rates, steps, subjects = read_data(use_prepared=False)
    subject_collector = SubjectCollector(subjects)

    final_train_path = sys.path[0] + "/final_train.pickle"
    if os.path.exists(final_train_path):
        with open(final_train_path, 'rb') as f:
            train = pickle.load(f)
        # train.train()
        # train.evaluate()
    else:
        # else I will have to do the training with minor parameter otimization

        # NOTE I have noticed that in some cases the motion feature have a lot
        # of missing data, so I will try to train for 3 different cases

        # train with using ALL of the prepared DATA --- missing motions are
        # replaced with 0
        train = Train(subject_collector.generate_sleep_data())
        train.train()
        train.evaluate()
        train.dump("w_original_motion_lstm.pickle")

        # train with using ALL FEATURE --- I cut the rows where motion values
        # started to miss
        train_w_cut_motion = Train(
            subject_collector.generate_sleep_data_w_cut("motion"))
        train_w_cut_motion.train()
        train_w_cut_motion.evaluate()
        train_w_cut_motion.dump("w_cut_motion_lstm.pickle")

        # selecting the best training and pickleing it to save time
        accuracies = [
            train.accuracy,
            train_w_cut_motion.accuracy]
        
        trainings = [train, train_w_cut_motion]
        train = trainings[accuracies.index(max(accuracies))]
        train.dump("final_train.pickle")

    # just curious which was the best
    get_train_vals(['w_original_motion_lstm.pickle', 'w_cut_motion_lstm.pickle'])
    # get_train_vals(['w_cut_motion_lstm.pickle'])
    
    
    print('Test accuracy was:', train.accuracy)
    print('Test loss was:', train.loss)
    
    # exit()
    data = subject_collector.generate_sleep_data()
    check_actual_vs_predicted(data)
    
    
    
    # NOTE Usage to predict based on new data
    # train.predict_sleep_phase(heart_rate, motion, step)
