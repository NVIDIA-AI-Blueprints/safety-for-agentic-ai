# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The script is for vLLM 0.8.5
# For vLLM 0.9.0 or 0.9.1, you need to comment out the Qwen3ReasoningParser import and __all__ entry
cp parsers/custom_reasoning_parser.py /usr/local/lib/python3.12/dist-packages/vllm/reasoning/
echo """# SPDX-License-Identifier: Apache-2.0

from .abs_reasoning_parsers import ReasoningParser, ReasoningParserManager
from .deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from .granite_reasoning_parser import GraniteReasoningParser
from .qwen3_reasoning_parser import Qwen3ReasoningParser
from .custom_reasoning_parser import CustomReasoningParser

__all__ = [
    "ReasoningParser",
    "ReasoningParserManager",
    "DeepSeekR1ReasoningParser",
    "GraniteReasoningParser",
    "Qwen3ReasoningParser",
    "CustomReasoningParser",
]""" | tee /usr/local/lib/python3.12/dist-packages/vllm/reasoning/__init__.py