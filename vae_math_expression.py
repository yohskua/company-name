from rnn_vae import RnnVae
import operator
import numpy as np

def math_expressions_generation(n_samples=1000, n_digits=3, invert=True):
    operations, results = [], []
    math_operators = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "%": operator.mod,
    }
    for i in range(n_samples):
        a, b = np.random.randint(1, 10 ** n_digits, size=2)
        op = np.random.choice(list(math_operators.keys()))
        res = math_operators[op](a, b)
        operation = "".join([str(elem) for elem in (a, op, b)])
        if invert is True:
            operation = operation[::-1]
        result = "{:.5f}".format(res) if isinstance(res, float) else str(res)
        operations.append(operation)
        results.append(result)
    return operations, results

if __name__ == "main":
    quick_for_debugg = False
    n_samples = 100 if quick_for_debugg else int(1e5)
    X = []
    operations, results = math_expressions_generation(n_samples=n_samples, n_digits=3, invert=False)
    for operation, result in list(zip(operations, results))[:]:
        X.append(operation + "=" + result)

    for mathematical_expression in X[:20]:
        print(mathematical_expression)

    rnn_vae = RnnVae(X, latent_dim=64, intermediate_dim=64)
    rnn_vae.train(epochs=1, batch_size=16)
    #rnn_vae.model([K.constant(X_tr[:4]), K.constant(X_te[:4])])
    rnn_vae.X_in_encoder[:20]
    for _ in range(20):
        print(rnn_vae.sample_prediction())
