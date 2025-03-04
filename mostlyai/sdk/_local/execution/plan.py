# Copyright 2025 MOSTLY AI
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

from pydantic import BaseModel, Field, ConfigDict
from collections import deque
from mostlyai.sdk.domain import ModelEncodingType, Generator, SourceTable, StepCode, TaskType, ModelType, \
    SyntheticDataset
import uuid

TABULAR_MODEL_ENCODING_TYPES = [v for v in ModelEncodingType if v.startswith(ModelType.tabular)] + [
    ModelEncodingType.auto
]
LANGUAGE_MODEL_ENCODING_TYPES = [v for v in ModelEncodingType if v.startswith(ModelType.language)]

TRAINING_TASK_STEPS: list[StepCode] = [
    StepCode.pull_training_data,
    StepCode.analyze_training_data,
    StepCode.encode_training_data,
    StepCode.train_model,
]
TRAINING_TASK_STEPS_REPORT: list[StepCode] = [
    StepCode.generate_model_report_data,
    StepCode.create_model_report,
]
FINALIZE_TRAINING_TASK_STEPS: list[StepCode] = [
    StepCode.finalize_training,
]
GENERATION_TASK_STEPS: list[StepCode] = [
    StepCode.generate_data,
]
GENERATION_TASK_STEPS_REPORTS: list[StepCode] = [
    StepCode.create_data_report,
]
FINALIZE_GENERATION_TASK_STEPS: list[StepCode] = [
    StepCode.finalize_generation,
    StepCode.deliver_data,
]
PROBING_TASK_STEPS: list[StepCode] = [
    StepCode.generate_data,
]
FINALIZE_PROBING_TASK_STEPS: list[StepCode] = [
    StepCode.finalize_probing,
]

def _get_keys(table: SourceTable) -> list[str]:
    keys = [fk.column for fk in (table.foreign_keys or [])]
    if table.primary_key:
        keys.append(table.primary_key)
    return keys


def has_tabular_model(table: SourceTable) -> bool:
    return table.tabular_model_configuration is not None


def has_language_model(table: SourceTable) -> bool:
    return table.language_model_configuration is not None


class Task(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    parent_task_id: str | None = Field(None, alias="parentTaskId", title="Parent Task Id")
    target_table_name: str | None = Field(None, alias="targetTableName", title="Target Table Name")
    type: TaskType
    steps: list[StepCode] | None


class ExecutionPlan(BaseModel):
    tasks: list[Task]

    def add_task(self, task_type: TaskType, parent: Task | None = None, target_table_name: str | None = None, create_report: bool = False) -> Task:
        def get_steps(task_type: TaskType) -> list[StepCode] | None:
            match task_type:
                case TaskType.train_tabular | TaskType.train_language:
                    return TRAINING_TASK_STEPS + (TRAINING_TASK_STEPS_REPORT if create_report else [])
                case TaskType.finalize_training:
                    return FINALIZE_TRAINING_TASK_STEPS
                case TaskType.generate_tabular | TaskType.generate_language:
                    return GENERATION_TASK_STEPS + (GENERATION_TASK_STEPS_REPORTS if create_report else [])
                case TaskType.finalize_generation:
                    return FINALIZE_GENERATION_TASK_STEPS
                case TaskType.probe_tabular | TaskType.probe_language:
                    return PROBING_TASK_STEPS
                case TaskType.finalize_probing:
                    return FINALIZE_PROBING_TASK_STEPS
                case _:
                    return None

        task = Task(
            id=str(uuid.uuid4()),
            type=task_type,
            parent_task_id=parent.id if parent else None,
            target_table_name=target_table_name,
            steps=get_steps(task_type),
        )
        self.tasks.append(task)
        return task


def make_generator_execution_plan(generator: Generator) -> ExecutionPlan:
    execution_plan = ExecutionPlan(tasks=[])
    sync_task = execution_plan.add_task(TaskType.sync)
    for table in generator.tables:
        if has_tabular_model(table):
            execution_plan.add_task(TaskType.train_tabular, parent=sync_task, target_table_name=table.name, create_report=table.tabular_model_configuration.enable_model_report)
        if has_language_model(table):
            execution_plan.add_task(TaskType.train_language, parent=sync_task, target_table_name=table.name, create_report=table.language_model_configuration.enable_model_report)
    execution_plan.add_task(TaskType.sync)
    # post_training_sync = execution_plan.add_task(TaskType.sync)
    # finalize_task = execution_plan.add_task(TaskType.finalize_training, parent=post_training_sync)
    # execution_plan.add_task(TaskType.sync, parent=finalize_task)
    return execution_plan


def make_synthetic_dataset_execution_plan(synthetic_dataset: SyntheticDataset, generator: Generator, is_probe: bool = False) -> ExecutionPlan:
    execution_plan = ExecutionPlan(tasks=[])
    data_reports_map = {table.name: table.configuration.enable_data_report for table in synthetic_dataset.tables}
    generate_tabular_task_type = TaskType.probe_tabular if is_probe else TaskType.generate_tabular
    generate_language_task_type = TaskType.probe_language if is_probe else TaskType.generate_language
    finalize_task_type = TaskType.finalize_probing if is_probe else TaskType.finalize_generation
    sync_task = execution_plan.add_task(TaskType.sync)
    root_tables = [table for table in generator.tables if not any(fk.is_context for fk in table.foreign_keys or [])]

    def table_has_context_relation_to(table: SourceTable, parent_table_name: str) -> bool:
        if table.foreign_keys is not None and any(
            foreign_key.referenced_table == parent_table_name and foreign_key.is_context
            for foreign_key in table.foreign_keys
        ):
            return True
        return False

    # Process each root table and its subtree
    for root_table in sorted(root_tables, key=lambda t: t.name):
        if has_tabular_model(root_table):
            generate_task = execution_plan.add_task(
                generate_tabular_task_type, parent=sync_task, target_table_name=root_table.name, create_report=data_reports_map[root_table.name]
            )
        else:
            generate_task = sync_task
        if has_language_model(root_table):
            generate_task = execution_plan.add_task(
                generate_language_task_type, parent=generate_task, target_table_name=root_table.name, create_report=data_reports_map[root_table.name]
            )

        # Traverse child tables in tree level order traversal
        queue = deque([(root_table.name, generate_task)])
        while queue:
            current_table_name, parent_task = queue.popleft()

            # Find child tables
            child_tables = sorted(
                [table for table in generator.tables if table_has_context_relation_to(table, current_table_name)],
                key=lambda t: t.name,
            )
            # Process each child table
            for child_table in child_tables:
                generate_task = parent_task
                if has_tabular_model(child_table):
                    generate_task = execution_plan.add_task(
                        generate_tabular_task_type, parent=generate_task, target_table_name=child_table.name, create_report=data_reports_map[child_table.name]
                    )
                if has_language_model(child_table):
                    generate_task = execution_plan.add_task(
                        generate_language_task_type, parent=generate_task, target_table_name=child_table.name, create_report=data_reports_map[child_table.name]
                    )

                queue.append((child_table.name, generate_task))

    post_generation_sync_task = execution_plan.add_task(TaskType.sync)
    finalize_task = execution_plan.add_task(finalize_task_type, parent=post_generation_sync_task)
    execution_plan.add_task(TaskType.sync, parent=finalize_task)
    return execution_plan
