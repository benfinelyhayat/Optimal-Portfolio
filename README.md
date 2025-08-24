# Optimal-Portfolio
Using MPT to create a portfolio; this is for port opt demonstration so will use basic methods for sample mean and sd but can easily incorporate methods such as rolling means and such to the model with minimal tweaking.
this is just a demonstration so will use pseudo stats. 
## N risky assets, with return given.
https://quantmathreadme.blogspot.com/2025/08/optimal-portfolio-theory.html

```
import numpy as np
individual_sd_matrix = np.array([
    [0.01, 0, 0, 0],
    [0, 0.2, 0, 0],
    [0 ,0 ,0.2 ,0],
    [0, 0, 0, 0.3]
])
Corr_Matrix = np.array([
    [0, 0.4, 0.3, 0.3],
    [0.4, 0, 0.2, 0.4],
    [0.3, 0.2, 0, 0.5],
    [0.3, 0.4, 0.5, 0]
]) + np.identity(4)

Expected_matrix = np.array([
    [0.05],
    [0.07],
    [0.1],
    [0.2]
])
Sigma = individual_sd_matrix @ Corr_Matrix @ individual_sd_matrix

columb_1s = np.array([
    [1],
    [1],
    [1],
    [1]
])
A = columb_1s.T @ np.linalg.inv(Sigma) @ columb_1s
B = Expected_matrix.T @ np.linalg.inv(Sigma) @ columb_1s
C = Expected_matrix.T @ np.linalg.inv(Sigma) @ Expected_matrix

alphebet_matrix = np.linalg.inv(np.array([
    [B.item() , A.item() ],
    [C.item() , B.item() ]
]))
idk = np.array([
    [1],
    [0.07]
])
parameter_solutions =  alphebet_matrix @ idk

w_star = parameter_solutions[0,0].item() * np.linalg.inv(Sigma) @ Expected_matrix + parameter_solutions[1,0].item() * np.linalg.inv(Sigma) @ columb_1s

w_star_renormalised = w_star /w_star.flatten().sum() #floating point error occurs without

print(w_star_renormalised)
```
