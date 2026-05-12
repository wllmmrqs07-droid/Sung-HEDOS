import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os

# Fixed Paramters (Table 1)
a = 0.01 #tumor growth
f = 0.033 #Lymphocyte decay rate
r = 0.14 #Inactivated tumor cell decay rate

#Fitted Paramaters (Table 1)
omega1 = 0.119 #Tumor-directed lymphocyte efficiency
omega2 = 0.003 #Tumor recruitment constant
omega3 = 0.009 #Inactive tumor recruitment constant
g = 7.33 * 10**10 #Half saturation constant
s = 1.47 * 10**8 #Lymphocyte regeneration
alpha_T = 0.139 #Tumor - LQ cell kill
beta_T = alpha_T / 14.3 #beta_t defined with respect to alpha_t
alpha_L = 0.737 #Lymphocytes - LQ cell kill

def odes(t, y):
    T, I, M, L = y

    #Equation a
    dT = ( a*T - omega1*T*L / (g+T+M) )

    #Equation b
    dI = (-r*I )

    #Equation c
    dM = ( a*M - omega1*M*L / (g+T+M) )

    #Equation d
    dL = ( (omega2*(T+M)*L / (g+T+M)) + L*(omega3*I / (g+I)) + s - f*L )

    return [dT, dI, dM, dL]

#DVH Doses and volume_fraction should be found from HEDOS output
def radiation(T, I, M, L, dose_T, DVH_doses, volume_fraction):
    #linear-quadratic cell kill
    LQ = np.exp(-alpha_T * dose_T - beta_T * dose_T**2)
    
    T_new = T * LQ #Equation e
    I_new = I + T * (1 - LQ) #Equation f
    M_new = M 

    L_new = np.sum(volume_fraction * L * np.exp(-alpha_L * DVH_doses)) #Equation g

    return T_new, I_new, M_new, L_new



def simulate(T0, I0, M0, L0, fraction_times, D_T_per_fraction,
             DVH_doses, volume_fraction, t_end):
  
    print("T after 15 fractions (radiation only):", T0 * (0.5052**15))
    print("Normalized:", 0.5052**15)
    y = [T0, I0, M0, L0]
    t_current = 0.0
    
    t_out, y_out = [t_current], [y]
    
    for t_frac in fraction_times:
        #Integrate from current time up to just before fraction
        if t_frac > t_current:
            t_span = (t_current, t_frac)
            t_eval = np.linspace(t_current, t_frac, max(2, int((t_frac - t_current)*10)))
            sol = sp.integrate.solve_ivp(odes, t_span, y, t_eval=t_eval,
                            method='RK45', rtol=1e-8, atol=1e-8)
            for i in range(len(sol.t)):
                t_out.append(sol.t[i])
                y_out.append(sol.y[:, i].tolist())
            y = sol.y[:, -1].tolist()
        
        # Apply radiation
        y[0], y[1], y[2], y[3] = radiation(
            y[0], y[1], y[2], y[3], D_T_per_fraction, DVH_doses, volume_fraction)
        t_out.append(t_frac)
        y_out.append(y[:])
        t_current = t_frac
    
    # Integrate remainder after last fraction
    if t_end > t_current:
        sol = sp.integrate.solve_ivp(odes, (t_current, t_end), y,
                        t_eval=np.linspace(t_current, t_end, 1000),
                        method='RK45', rtol=1e-8, atol=1e-9)
        for i in range(len(sol.t)):
            t_out.append(sol.t[i])
            y_out.append(sol.y[:, i].tolist())
    
    return np.array(t_out), np.array(y_out)

T0 = 1.07e11    #Cells (10^9/cm^3 * 107 cm^3 GTV)
I0 = 0.0
M0 = 1.07e8     #0.1% of primary
L0 = 5.61e9     #1122/mm^3 * 5L blood volume

#15 fractions, weekdays only
fraction_times = [d for d in range(0, 21) if d % 7 not in [5, 6]][:15]
D_T = 58.0 / 15  #~3.87 Gy per fraction

dir_path = os.path.dirname(os.path.abspath(__file__))

DVH_doses       = np.load(os.path.join(dir_path, 'DVH_doses.npy'))
volume_fraction = np.load(os.path.join(dir_path, 'DVH_volume_fractions.npy'))

print("Dose per fraction to tumor:", D_T)
print("Per-fraction tumor survival fraction:", 
      np.exp(-alpha_T * D_T - beta_T * D_T**2))

t_out, y_out = simulate(T0, I0, M0, L0, fraction_times, D_T,
                        DVH_doses, volume_fraction, t_end=250)

# Normalize to baseline values (as in Fig. 3 of the paper)
T_out = y_out[:, 0] / T0
I_out = y_out[:, 1] / T0  # normalized to initial tumor, same as paper
M_out = y_out[:, 2] / T0
L_out = y_out[:, 3] / L0  # normalized to baseline lymphocyte count

# Shade the treatment period
t_treatment_start = fraction_times[0]
t_treatment_end   = fraction_times[-1]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# --- Top panel: tumor compartments ---
ax1.plot(t_out, T_out, label='Primary tumor (T)', color='red')
ax1.plot(t_out, I_out, label='Inactivated tumor (I)', color='orange')
ax1.plot(t_out, M_out, label='Metastatic tumor (M)', color='darkred', linestyle='--')
ax1.axvspan(t_treatment_start, t_treatment_end, alpha=0.15, color='yellow',
            label='Treatment period')
ax1.set_ylabel('Normalized cell count')
ax1.set_title('Tumor compartments')
ax1.legend()
ax1.grid(True)

# --- Bottom panel: lymphocytes ---
ax2.plot(t_out, L_out, label='Circulating lymphocytes (L)', color='blue')
ax2.axhline(y=0.7, color='grey', linestyle=':', label='70% recovery threshold')
ax2.axvspan(t_treatment_start, t_treatment_end, alpha=0.15, color='yellow',
            label='Treatment period')
ax2.set_ylabel('Normalized cell count')
ax2.set_xlabel('Time (days)')
ax2.set_title('Circulating lymphocytes')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()