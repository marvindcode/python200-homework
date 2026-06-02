# project_08.py
# Run this in Azure Cloud Shell after completing the Cost Analysis above.

# Fill in the hourly rates from your two Pricing Calculator estimates.
rate_a = 0.009    # Standard_B1s hourly rate (Scenario A)
rate_b = 3.060    # Standard_NC6s_v3 hourly rate (Scenario B, VM only)

hours_a = 160   # Scenario A: 8h/day, 5 days/week, ~4 weeks
hours_b = 730   # Scenario B: always on

cost_a = rate_a * hours_a
cost_b = rate_b * hours_b

print("=== Monthly Cost Estimates ===")
print(f"Scenario A (lightweight):       ${cost_a:.2f}")
print(f"Scenario B (GPU VM only):       ${cost_b:.2f}")

if cost_a > 0:
    print(f"Scenario B VM costs {cost_b / cost_a:.1f}x more than Scenario A") 




# Write-up:
# For scenario A, I chose the Standard_B1s VM, which is a cost-effective option for lightweight workloads. The hourly rate for this VM is $0.009, and I estimated that it would be used for 160 hours per month (8 hours/day, 5 days/week). This results in a monthly cost of approximately $1.44.  The result of this scenario is very inexpensive ans is suitable for light workloads, the VM is small and only runs part time.
    
# For scenario B, I selected the Standard_NC6s_v3 VM, which is designed for GPU-intensive workloads. The hourly rate for this VM is higher compared with Scenario A and is significantly higher at $3.060. Since this VM is intended to be always on, I estimated it would be used for 730 hours per month (24/7). The total per month cost is $2,233.80.  This is scenario is more costly because it runs continuosly and has a much higher hourly rate due to the GPU capabilities.

# It was good to compare two scenarios, one with a lightweight VM and another with a GPU VM, to see the significant cost difference between them.  We can see how the prioces scalate significantly depending on the machine capabilities needed  for certain tasks.  Is important to identify the right VM, based on the workload to avoid overspending.  Cloud costs can increase significantly if the wrong VM is chosen, especially if it runs continuously.
 
#  It was interesting to see the Azure price calculator, to compare the costs of different VMs and understand how the pricien works. The calculator provided a clear breakdown of the costs, which helped me to make informed decisions about which VM to choose for each scenario. It also highlighted the importance of considering both the hourly rate and the usage hours when estimating costs.

 