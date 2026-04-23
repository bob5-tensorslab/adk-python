# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A2A specific experimental decorator with custom warning message."""

from __future__ import annotations

from typing import Any

from google.adk.utils.feature_decorator import _make_feature_decorator


def a2a_experimental(message_or_obj: Any = None) -> Any:
    if message_or_obj is not None and (
        isinstance(message_or_obj, type) or callable(message_or_obj)
    ):
        return message_or_obj
    return lambda obj: obj

"""Mark a class or function as experimental A2A feature.

This decorator shows a specific warning message for A2A functionality,
indicating that the API is experimental and subject to breaking changes.

Sample usage:

```
# Use with default A2A experimental message
@a2a_experimental
class A2AExperimentalClass:
  pass

# Use with custom message (overrides default A2A message)
@a2a_experimental("Custom A2A experimental message.")
def a2a_experimental_function():
  pass

# Use with empty parentheses (same as default A2A message)
@a2a_experimental()
class AnotherA2AClass:
  pass
```
"""
