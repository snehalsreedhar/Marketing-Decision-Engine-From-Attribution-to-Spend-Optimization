from scripts.train_all import run_all

results = run_all()
results["alignment"].to_csv("outputs/cross_model_alignment.csv", index=False)
