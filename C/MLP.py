import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ---------------- Step 2: input, target, learning rate, weights, biases ----------------
x1, x2, x3 = 1, 1, 0
target = 1
lr = 0.6

# input -> hidden
w14, w24, w34 = 0.2, -0.1, 0.3
w15, w25, w35 = 0.3, 0.2, -0.4
# hidden -> output
w46, w56 = -0.3, 0.1
# biases
b4, b5, b6 = 0.1, 0.3, -0.3

print("Initial weights:")
print(f"  w14={w14}, w24={w24}, w34={w34}")
print(f"  w15={w15}, w25={w25}, w35={w35}")
print(f"  w46={w46}, w56={w56}")
print(f"Initial biases: b4={b4}, b5={b5}, b6={b6}")

# ---------------- Step 3: forward propagation ----------------
net4 = w14*x1 + w24*x2 + w34*x3 + b4
net5 = w15*x1 + w25*x2 + w35*x3 + b5
o4, o5 = sigmoid(net4), sigmoid(net5)

net6 = w46*o4 + w56*o5 + b6
o6 = sigmoid(net6)

print(f"\nNet input of unit 4 = {net4:.4f}   Output O4 = {o4:.4f}")
print(f"Net input of unit 5 = {net5:.4f}   Output O5 = {o5:.4f}")
print(f"Net input of unit 6 = {net6:.4f}   Output O6 = {o6:.4f}")
o6_before = o6
print(f"Desired output = {target}, Error = {target - o6:.4f}")

# ---------------- Step 4: backward propagation of error ----------------
err6 = o6 * (1 - o6) * (target - o6)
err4 = o4 * (1 - o4) * err6 * w46      # uses old w46
err5 = o5 * (1 - o5) * err6 * w56      # uses old w56

print(f"\nError at unit 6 (Err6) = {err6:.6f}")
print(f"Error at unit 5 (Err5) = {err5:.6f}")
print(f"Error at unit 4 (Err4) = {err4:.6f}")

# ---------------- Step 5: update weights and biases ----------------
old = dict(w14=w14, w15=w15, w24=w24, w25=w25, w34=w34, w35=w35,
           w46=w46, w56=w56, b4=b4, b5=b5, b6=b6)

# hidden -> output weights
w46 += lr * err6 * o4
w56 += lr * err6 * o5
# input -> hidden weights
w14 += lr * err4 * x1; w24 += lr * err4 * x2; w34 += lr * err4 * x3
w15 += lr * err5 * x1; w25 += lr * err5 * x2; w35 += lr * err5 * x3
# biases
b6 += lr * err6
b5 += lr * err5
b4 += lr * err4

new = dict(w14=w14, w15=w15, w24=w24, w25=w25, w34=w34, w35=w35,
           w46=w46, w56=w56, b4=b4, b5=b5, b6=b6)

print(f"\n{'Param':<6}{'Old value':>12}{'New value':>12}")
print("-" * 30)
for k in old:
    print(f"{k:<6}{old[k]:>12.4f}{new[k]:>12.4f}")

# ---------------- Step 6: forward pass with updated weights ----------------
o4 = sigmoid(w14*x1 + w24*x2 + w34*x3 + b4)
o5 = sigmoid(w15*x1 + w25*x2 + w35*x3 + b5)
o6_new = sigmoid(w46*o4 + w56*o5 + b6)

print(f"\nOutput before update : {o6_before:.4f}")
print(f"Output after update  : {o6_new:.4f}  (desired = {target})")

# ---------------- Step 7: train until output is within 0.01 of target ----------------
W_h = np.array([[0.2, 0.3],      # X1 -> (4, 5)
                [-0.1, 0.2],     # X2 -> (4, 5)
                [0.3, -0.4]])    # X3 -> (4, 5)
b_h = np.array([0.1, 0.3])
W_o = np.array([-0.3, 0.1])      # (4, 5) -> 6
b_o = -0.3
x = np.array([1, 1, 0])

epoch = 0
while True:
    epoch += 1
    # forward
    h = sigmoid(x @ W_h + b_h)
    out = sigmoid(h @ W_o + b_o)
    # backward
    d_out = out * (1 - out) * (target - out)
    d_h = h * (1 - h) * d_out * W_o
    # update
    W_o += lr * d_out * h
    b_o += lr * d_out
    W_h += lr * np.outer(x, d_h)
    b_h += lr * d_h

    if epoch in (1, 10, 100, 500, 1000):
        print(f"Epoch {epoch:5d}  output = {out:.4f}  error = {target - out:.4f}")
    if abs(target - out) < 0.01:
        break

print(f"\nConverged after {epoch} epochs, output = {out:.4f}")
print("Final input->hidden weights:\n", np.round(W_h, 4))
print("Final hidden biases :", np.round(b_h, 4))
print("Final hidden->output weights:", np.round(W_o, 4))
print("Final output bias :", round(float(b_o), 4))