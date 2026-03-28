from utils.metrics import compare, component_analysis
import utils.plot as p

compare()
component_analysis()

print("\n=== ANALYSIS START ===\n")
p.plot_loss()
p.plot_smooth()
p.plot_bar()
p.plot_component()
print("\n=== ANALYSIS COMPLETE ===\n")