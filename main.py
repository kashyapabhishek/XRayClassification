from training_model.custom_ann import CustomAnn
from training_model.custom_cnn import CustomCnn
from training_model.transfer_learning import TransferLearning
import os

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    obj = CustomAnn()
    obj_cnn = CustomCnn()
    obj_transfer = TransferLearning(open(os.path.join(os.getcwd(), 'logs/train.txt'), 'w+'))
    #obj_transfer.evaluate()
    obj_transfer.predict(os.path.join(os.getcwd(), 'data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg'))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
