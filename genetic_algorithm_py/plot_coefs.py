import matplotlib.pyplot as plt
def evaluate_function(x,coefs):
        value = 0.0
        x_pow = 1
        for coef in coefs:
            value += coef * x_pow
            x_pow *= x
        return value
    
def plot_coefs(single_organism_coefs):
    x = 0.0
    x_values = []
    function_values = []
    step = 0.03
    while x < 10:
        function_values.append(evaluate_function(x,single_organism_coefs))
        x_values.append(x)
        x += step
    plt.plot(x_values,function_values)
