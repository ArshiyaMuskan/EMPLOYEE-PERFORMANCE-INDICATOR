import os

print("🚀 Starting Employee Performance Project Pipeline...\n")

# Step 1: Data Generation
print("📊 Generating dataset...")
os.system("python src/1_data_generation.py")

# Step 2: Preprocessing
print("\n🧹 Preprocessing data...")
os.system("python src/2_preprocessing.py")

# Step 3: EDA
print("\n📈 Running EDA...")
os.system("python src/3_eda.py")

# Step 4: Model Training
print("\n🤖 Training model...")
os.system("python src/4_model.py")

# Step 5: Prediction
print("\n🔮 Running prediction...")
os.system("python src/5_predict.py")

print("\n✅ ALL STEPS COMPLETED SUCCESSFULLY!")