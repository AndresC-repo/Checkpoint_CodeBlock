# After system crash or interruption, just re-run the same script:

# This will automatically:
# 1. Detect existing checkpoints
# 2. Reload model, optimizer, and scheduler states
# 3. Continue from the next epoch
# 4. Maintain all previous best metrics

tm = create_training_manager(
    model=model,
    optimizer=optimizer, 
    model_name="my_model"  # Same name as before
)

print(f"Resumed from epoch {tm.current_epoch}")
print(f"Previous best: {tm.get_best_metric_value('accuracy')}")

# Continue training seamlessly
for epoch in range(tm.current_epoch, 100):
    # ... training code
    tm.save(metrics)