import pickle
import matplotlib.pyplot as plt

ret = pickle.load(open('losses.pkl', 'rb'))

for losses in ret[False]:
    plt.plot(losses, 'r')
for losses in ret[True]:
    plt.plot(losses, 'g')

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Red = no TOI, Green = TOI')
plt.show()
