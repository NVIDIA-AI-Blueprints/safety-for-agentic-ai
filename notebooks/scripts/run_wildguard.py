export JUDGE_API_KEY="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"
safety-eval --model-name "test-model" \
            --model-url http://localhost:5001/v1 \
            --judge-url https://d27e7649-daf1-46c2-ba22-49374402c31d.invocation.api.nvcf.nvidia.com/v1 \
            --results-dir ./wildguard_results \
            --concurrency 64 \
            --eval wildguard \
            --inference_params "temperature=0.6,top_p=0.95,max_completion_tokens=12000" &> "./wildguard_results/safety-eval-wildguard.log"
