import os
import numpy as np 

# Find unique true labels
def get_true_labels(input_file, classes):

    classes_label = classes
    single_recording_labels=np.zeros(len(classes),dtype=int)


    with open(input_file,'r') as f:
        first_line = f.readline()
        recording_label=first_line.split(' ')[0]
        # print(recording_label)
        for lines in f:
            if lines.startswith('#Dx'):
                tmp = lines.split(': ')[1].split(',')
                for c in tmp:
                    idx = classes.index(c.strip())
                    single_recording_labels[idx]=1

    return recording_label,classes_label,single_recording_labels
# Find unique number of classes
def get_classes(files):

    classes=set()
    for input_file in files:
        with open(input_file,'r') as f:
            for lines in f:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())

    return sorted(classes)

label_directory = 'input/PhysioNetChallenge2020_Training_CPSC/Training_WFDB'
labels=[]

# Find label and output files.
label_files = []
for i,f in enumerate(sorted(os.listdir(label_directory))):
    if i>100:
        break
    g = os.path.join(label_directory, f)
    if os.path.isfile(g) and not f.lower().startswith('.') and f.lower().endswith('hea'):
        label_files.append(g)
label_files = sorted(label_files)

classes = get_classes(label_files)

# Load labels and outputs.
num_files = len(label_files)

for k in range(num_files):

    recording_label,classes_label,single_recording_labels=get_true_labels(label_files[k], classes)

    labels.append(single_recording_labels)

labels=np.array(labels)
print(labels)
