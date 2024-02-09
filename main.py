import pickle
import sys
import os

from feature import Motion, HeartRate, Step
from label import Label
from data_handler import read_files
from subject import Subject, SubjectCollector
from train import Train


def read_wo_prepared():
    labels_dict = read_files("labels")
    labels = []
    motions = []
    heart_rates = []
    steps = []
    subjects = []
    for key, value in labels_dict.items():
        print(">>> ", key)
        label = Label(key, value)
        labels.append(label)
        motions.append(
            Motion(
                key,
                read_files(
                    "motions",
                    idx=key),
                label.start_time,
                label.end_time,
                label.prepared_data))
        # motions[-1].normalize()
        steps.append(
            Step(
                key,
                read_files(
                    "steps",
                    idx=key),
                label.start_time,
                label.end_time,
                label.prepared_data))
        steps[-1]
        heart_rates.append(
            HeartRate(
                key,
                read_files(
                    "heart_rates",
                    idx=key),
                label.start_time,
                label.end_time,
                label.prepared_data))
        heart_rates[-1]
        s = Subject(labels[-1], motions[-1], heart_rates[-1], steps[-1])
        s.plot()
        s.save_normalized_data()
        subjects.append(s)
        print("DONE")
    return labels, motions, heart_rates, steps, subjects


def read_w_prepared():
    labels_dict = read_files("labels")
    labels = []
    motions = []
    heart_rates = []
    steps = []
    subjects = []
    for key, value in labels_dict.items():
        print(">>> ", key)
        label = Label(key, value)
        labels.append(label)
        motions.append(
            Motion(
                key,
                read_files(
                    "motions",
                    idx=key,
                    use_prepared=True),
                label.start_time,
                label.end_time,
                label.prepared_data,
                prepared_used=True))
        # motions[-1].normalize()
        steps.append(
            Step(
                key,
                read_files(
                    "steps",
                    idx=key,
                    use_prepared=True),
                label.start_time,
                label.end_time,
                label.prepared_data,
                prepared_used=True))
        steps[-1]
        heart_rates.append(
            HeartRate(
                key,
                read_files(
                    "heart_rates",
                    idx=key,
                    use_prepared=True),
                label.start_time,
                label.end_time,
                label.prepared_data,
                prepared_used=True))
        heart_rates[-1]
        s = Subject(labels[-1], motions[-1], heart_rates[-1], steps[-1])
        s.plot()
        # s.save_normalized_data()
        subjects.append(s)
        print("DONE")
    return labels, motions, heart_rates, steps, subjects


def get_train_vals(filenames):
    for fname in filenames:
        with open(fname, 'rb') as f:
            train = pickle.load(f)
            print('--------- ', fname, ' ---------')
            print(train.loss, train.accuracy)


def check_actual_vs_predicted(data):
    for i in range(len(data["heart_rate"])):
        heart_rate = data['heart_rate'][i]
        # print(heart_rate)
        motion = data['motion'][i]
        # print(motion)
        step = data['step'][i]
        # print(step)
        print()
        print(
            data['sleep_phase'][i],
            "--- vs ---",
            train.predict_sleep_phase(
                heart_rate,
                motion,
                step))


if __name__ == "__main__":
    try:
        # Because preparing the data took a lot of time, I saved the prepared
        # data, so first of all I will try to load the already prepared data if
        # exists
        labels, motions, heart_rates, steps, subjects = read_w_prepared()
    except BaseException:
        # If it does not exist I will prepare the data here
        print("Prepared data not found")
        labels, motions, heart_rates, steps, subjects = read_wo_prepared()
    subject_collector = SubjectCollector(subjects)

    final_train_path = sys.path[0] + "/final_train.pickle"
    if os.path.exists(final_train_path):
        # if an 'ideal' train object was found I will use that
        with open(final_train_path, 'rb') as f:
            train = pickle.load(f)
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

        # train without using the motion feature
        train_wo_motion = Train(
            subject_collector.generate_sleep_data(),
            with_motion=False)
        train_wo_motion.train()
        train_wo_motion.evaluate()
        train_wo_motion.dump("wo_motion_lstm.pickle")

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
            train_wo_motion.accuracy,
            train_w_cut_motion.accuracy]
        trainings = [train, train_wo_motion, train_w_cut_motion]
        train = trainings[accuracies.index(max(accuracies))]
        train.dump("final_train.pickle")

    # just curious which was the best
    get_train_vals(['w_original_motion_lstm.pickle',
                   'wo_motion_lstm.pickle', 'w_cut_motion_lstm.pickle'])


    # # need to take a look at it, 'cause in previous cases always predicted 2 --- hope so it will change
    # # yeah, I know that is kinda sad that it does not predict right :(
    # data = subject_collector.generate_sleep_data()
    # check_actual_vs_predicted(data)
    
    
    
    
    # To predict based on new data
    # train.predict_sleep_phase(heart_rate, motion, step)


# NOTE Because it always predicts 2 I tried to apply different scalers
# StandardScaler - PREDICTS always 2
    # ---------  w_original_motion_lstm.pickle  ---------
    # 1.040042519569397 0.6349999904632568

    # ---------  wo_motion_lstm.pickle  ---------
    # 1.2114793062210083 0.5899999737739563

    # ---------  w_cut_motion_lstm.pickle  ---------
    # 0.9634985327720642 0.6449999809265137
    
    
# RobustScaler - PREDICTS always 2
    # ---------  w_original_motion_lstm.pickle  ---------
    # 1.0401597023010254 0.625

    # ---------  wo_motion_lstm.pickle  ---------
    # 1.1780474185943604 0.6050000190734863

    # ---------  w_cut_motion_lstm.pickle  ---------
    # 0.9153749346733093 0.6549999713897705
    
    
# MinMaxScaler - PREDICTS always 2
    # ---------  w_original_motion_lstm.pickle  ---------
    # 1.0553537607192993 0.625


    # ---------  w_original_motion_lstm.pickle  ---------
    # 1.0553537607192993 0.625


    # ---------  wo_motion_lstm.pickle  ---------
    # 1.208256483078003 0.574999988079071


# NOTE There were a lot of parameters which I could try to change, but I ran out of time 
