with the goal of optimizing for expected return function J with respect to the neural network theta:

Two neural networks are initilized:
* theta, a neural network taking input as s and returning the distribution of actions (mean is selected)
* phi, which approximates the value function of the policy

The env is ran some amount, from that we get rewards, which we use later

the advantage is estimated with the value function NN, phi

with the advantage we weight the log gradient of the policy, giving us the gradient of the returns approximation of the policy, using that approximation to ascent the policy NN

with the return calculated, we gradient descent with the mean squared error on the value function NN

