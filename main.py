from training_model.custom_ann import CustomAnn
from training_model.custom_cnn import CustomCnn
from training_model.transfer_learning import TransferLearning

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    obj = CustomAnn()
    obj_cnn = CustomCnn()
    obj_transfer = TransferLearning()
    obj_transfer.evaluate()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
