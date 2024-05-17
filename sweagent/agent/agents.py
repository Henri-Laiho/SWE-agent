import json
import os
import re
import logging
import subprocess

from dataclasses import dataclass
from pathlib import Path
from time import sleep

import vertexai
from google.api_core.exceptions import ResourceExhausted, InternalServerError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Output, Input
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from openai import RateLimitError
from simple_parsing.helpers.fields import field
from simple_parsing.helpers.serialization.serializable import FrozenSerializable
from simple_parsing.helpers.flatten import FlattenedAccess
from swebench import KEY_MODEL, KEY_INSTANCE_ID, KEY_PREDICTION

from aiderrepomap.swe_repomap import RepoMap
from sweagent.agent.commands import Command, ParseCommand
from sweagent.agent.history_processors import HistoryProcessor
from sweagent.agent.models import (
    APIStats,
    ContextWindowExceededError,
    CostLimitExceededError,
    ModelArguments,
    get_model,
)
from sweagent.agent.parsing import ParseFunction, FormatError
from sweagent.environment.utils import LOGGER_NAME
from sweagent.environment.swe_env import SWEEnv
from tenacity import RetryError
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class Subroutine(FrozenSerializable):
    name: str
    agent_file: str
    # one of "action", "observation", "response", "state", "thought"
    return_type: str = None  # type: ignore
    init_observation: Optional[str] = None
    end_name: Optional[str] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    model: Optional[ModelArguments] = None
    agent_args: Optional[Any] = None


@dataclass(frozen=True)
class AgentConfig(FrozenSerializable):
    system_template: str
    instance_template: str
    next_step_template: Optional[str] = None  # defaults to instance_template
    next_step_no_output_template: Optional[str] = None  # defaults to next_step_template
    strategy_template: Optional[str] = None
    demonstration_template: Optional[str] = None
    demonstrations: list[str] = field(default_factory=list)
    put_demos_in_history: bool = (
        False  # if True, add demonstration to history instead of as a single message
    )
    # defaults to format_error_template in ParseFunction
    format_error_template: str = None  # type: ignore
    command_files: list[str] = field(default_factory=list)
    env_variables: dict[str, str] = field(default_factory=dict)
    util_functions: list[str] = field(default_factory=list)
    submit_command: str = "submit"
    parse_function: str = "ThoughtActionParser"
    parse_command: str = "ParseCommandBash"
    history_processor: str = "DefaultHistoryProcessor"
    history_processor_args: dict[str, Any] = field(default_factory=dict)
    command_docs: str = None  # type: ignore
    blocklist_error_template: str = (
        "Interactive operation '{name}' is not supported by this environment"
    )
    blocklist: Tuple[str, ...] = (
        "vim",
        "vi",
        "emacs",
        "nano",
        "nohup",
        "git",
    )
    blocklist_standalone: Tuple[str, ...] = (
        "python",
        "python3",
        "ipython",
        "bash",
        "sh",
        "exit",
        "/bin/bash",
        "/bin/sh",
        "nohup",
        "vi",
        "vim",
        "emacs",
        "nano",
    )
    # Should extract environment state in a json readable form
    state_command: Command = Command(
        name="state",
        code="""state() {
            echo '{"working_dir": "'$(realpath --relative-to=$ROOT/.. $PWD)'"}';
        };""",
    )
    _commands: list[Command] = field(default_factory=list)
    _subroutines: dict[str, Subroutine] = field(default_factory=dict)
    subroutine_types: list[Subroutine] = field(default_factory=list)

    def __post_init__(self):
        if self.next_step_template is None:
            object.__setattr__(self, "next_step_template", self.instance_template)
        if self.next_step_no_output_template is None:
            object.__setattr__(
                self, "next_step_no_output_template", self.next_step_template
            )

        object.__setattr__(self, "parse_command", ParseCommand.get(self.parse_command))
        for file in self.command_files:
            commands = self.parse_command.parse_command_file(file)

            util_functions = [
                command for command in commands if command.name.startswith("_")
            ]
            commands = [
                command for command in commands if not command.name.startswith("_")
            ]

            object.__setattr__(
                self, "util_functions", self.util_functions + util_functions
            )
            object.__setattr__(self, "_commands", self._commands + commands)

        for subroutine in self.subroutine_types:
            if subroutine.name == "submit":
                raise ValueError("Cannot use 'submit' as a subroutine name")
            agent_args = AgentArguments(
                model=subroutine.model,
                config_file=subroutine.agent_file,
            )
            object.__setattr__(subroutine, "agent_args", agent_args)
            object.__setattr__(
                self, "_subroutines", {**self._subroutines, subroutine.name: subroutine}
            )

        multi_line_command_endings = {
            command.name: command.end_name
            for command in [*self._commands, *self._subroutines.values()]
            if command.end_name is not None
        }
        object.__setattr__(
            self, "multi_line_command_endings", multi_line_command_endings
        )
        object.__setattr__(
            self,
            "command_docs",
            self.parse_command.generate_command_docs(
                self._commands,
                self.subroutine_types,
                **self.env_variables,
            ),
        )
        object.__setattr__(
            self, "parse_function", ParseFunction.get(self.parse_function)
        )
        if self.format_error_template is None:
            object.__setattr__(
                self,
                "format_error_template",
                self.parse_function.format_error_template,
            )
        object.__setattr__(
            self,
            "format_error_template",
            self.format_error_template.format(**self.__dict__),
        )
        for command in self._commands:
            if command.name == self.submit_command:
                object.__setattr__(self, "submit_command_end_name", command.end_name)
                break
        object.__setattr__(
            self,
            "history_processor",
            HistoryProcessor.get(self.history_processor, **self.history_processor_args),
        )


@dataclass(frozen=True)
class AgentArguments(FlattenedAccess, FrozenSerializable):
    """Configure the agent's behaviour (templates, parse functions, blocklists, ...)."""
    model: ModelArguments = None

    # Policy can only be set via config yaml file from command line
    config_file: Optional[Path] = None
    config: Optional[AgentConfig] = field(default=None, cmd=False)

    def __post_init__(self):
        if self.config is None and self.config_file is not None:
            # If unassigned, we load the config from the file to store its contents with the overall arguments
            config = AgentConfig.load_yaml(self.config_file)
            object.__setattr__(self, "config", config)
        assert self.config is not None  # mypy
        for subroutine in getattr(self.config, "subroutines", {}).values():
            model_args = getattr(subroutine, "model")
            object.__setattr__(
                model_args,
                "per_instance_cost_limit",
                self.model.per_instance_cost_limit,
            )
            object.__setattr__(
                model_args, "total_cost_limit", self.model.total_cost_limit
            )


class Agent:
    """Agent handles the behaviour of the model and how it interacts with the environment."""

    def __init__(self, name: str, args: AgentArguments):
        self.name = name
        self.model = get_model(
            args.model, args.config._commands + args.config.subroutine_types
        )
        self.config = args.config
        assert self.config is not None  # mypy
        self.system_args = {
            "command_docs": self.config.command_docs,
            **self.config.env_variables,
        }
        self.instance_args = None
        self._parse_command_patterns()
        self.history = []
        self.last_container_id = None

    def setup(self, instance_args, init_model_stats=None) -> None:
        """Setup the agent for a new instance."""
        assert self.config is not None  # mypy
        self.model.reset_stats(init_model_stats)
        self.instance_args = instance_args

        system_msg = self.config.system_template.format(**self.system_args)
        logger.info(f"SYSTEM ({self.name})\n{system_msg}")

        self.history: List[Dict[str, Any]] = [
            {"role": "system", "content": system_msg, "agent": self.name},
        ]

        if len(self.config.demonstrations) > 0 and "history_to_messages" in dir(
                self.model
        ):
            for demonstration_path in self.config.demonstrations:
                if (
                        self.config.demonstration_template is None
                        and not self.config.put_demos_in_history
                ):
                    raise ValueError(
                        "Cannot use demonstrations without a demonstration template or put_demos_in_history=True"
                    )

                # Load history
                logger.info(f"DEMONSTRATION: {demonstration_path}")
                demo_history = json.load(open(demonstration_path, "r"))["history"]
                demo_history = [
                    entry
                    for entry in demo_history
                    if ("agent" not in entry)
                       or ("agent" in entry and entry["agent"] == "primary")
                ]

                if self.config.put_demos_in_history:
                    if self.config.demonstration_template is not None:
                        logger.warning(
                            "Demonstration template is ignored for put_demos_in_history=True"
                        )
                    # Add demonstration to history directly as separate messages
                    for entry in demo_history:
                        if entry["role"] != "system":
                            entry["is_demo"] = True
                            self.history.append(entry)
                else:
                    # Add demonstration as single message to history
                    demo_message = self.model.history_to_messages(
                        demo_history,
                        is_demonstration=True,
                    )
                    demonstration = self.config.demonstration_template.format(
                        **{"demonstration": demo_message}
                    )
                    self.history.append(
                        {
                            "agent": self.name,
                            "content": demonstration,
                            "is_demo": True,
                            "role": "user",
                        }
                    )

    def forget_messages(self, search_keywords: list[str], delete_count: int):
        kws = [x.lower() for x in search_keywords]
        to_delete = [(i, self.history[i]) for i in range(len(self.history)) if any(kw in self.history[i]['content'].lower() for kw in kws)]
        if abs(len(to_delete) - delete_count)/len(self.history) > 0.25:
            logger.warning('Memory manager tried to delete %s but wanted to delete %s messages (keywords: %s)' % (len(to_delete), delete_count, search_keywords))
            return

        deleted = ""
        for i, it in to_delete[::-1]:
            if i < 3:
                logger.warning('Memory manager trying to delete initial messages: %s; %s' % (delete_count, search_keywords))
                continue
            if i >= len(self.history) - 6:
                logger.warning('Memory manager trying to delete too recent memories: %s; %s' % (delete_count, search_keywords))
                continue
            if it['role'] == 'user':
                if self.history[i+1]['role'] == 'assistant':
                    self.history.pop(i)
                    self.history.pop(i-1)
            elif it['role'] == 'assistant':
                if self.history[i+1]['role'] == 'user':
                    self.history.pop(i+1)
                    self.history.pop(i)
            else:
                continue
            deleted += f"{it['content'][:min(40, len(it['content']))]}...\n\n"

        self.mc_trajectory.append(
            {
                "action": "N/A",
                "observation": f"Memory manager deleted {len(to_delete)} messages: {deleted}",
                "response": "N/A",
                "state": "",
                "thought": "N/A",
            }
        )


    @property
    def state_command(self) -> str:
        """Return the bash command that will be used to extract the environment state."""
        return self.config.state_command.name

    @property
    def local_history(self) -> list[dict[str, str]]:
        """Return the history of the agent since the last reset."""
        return self.config.history_processor(
            [entry for entry in self.history if entry["agent"] == self.name]
        )

    def save_trajectory(self, trajectory, traj_dir, env, info):
        log_path = traj_dir / (env.record["instance_id"] + ".traj")
        log_dict = {
            "environment": env.name,
            "trajectory": trajectory,
            "history": self.history,
            "info": info,
        }
        with log_path.open("w") as f:
            json.dump(log_dict, f, indent=2)
        logger.info(f"Saved trajectory to {log_path}")

    def _get_first_match(self, action: str, pattern_type: str) -> Optional[re.Match]:
        """Return the first match of a command pattern in the action string."""
        assert self.config is not None  # mypy
        if pattern_type == "subroutine":
            patterns = {k: v for k, v in self.subroutine_patterns.items()}
        elif pattern_type == "multi_line":
            patterns = {
                k: v
                for k, v in self.command_patterns.items()
                if k in self.config.multi_line_command_endings
                   or k == self.config.submit_command
            }
            patterns += {
                k: v
                for k, v in self.subroutine_patterns.items()
                if k in self.config.multi_line_command_endings
            }
        elif pattern_type == "multi_line_no_subroutines":
            patterns = {
                k: v
                for k, v in self.command_patterns.items()
                if k in self.config.multi_line_command_endings
            }
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        matches = list()
        for name, pat in patterns.items():
            match = pat.search(action)
            if match:
                matches.append(match)
        if len(matches) == 0:
            return None
        matches = sorted(matches, key=lambda x: x.start())
        return matches[0]

    def _guard_multiline_input(self, action: str) -> str:
        """Split action by multiline commands, then append the first line in each multiline command with "<< '{end_name}'".
        Multiline commands (which are specified by an end_name) are commands that span multiple lines and are terminated by a specific end_name.

        Their multi-line argument is sent using a heredoc, which is a way to send a multi-line string to a command in bash.
        """
        parsed_action = list()
        rem_action = action
        while rem_action.strip():
            first_match = self._get_first_match(rem_action, "multi_line_no_subroutines")
            if first_match:
                pre_action = rem_action[: first_match.start()]
                match_action = rem_action[first_match.start(): first_match.end()]
                rem_action = rem_action[first_match.end():]
                if pre_action.strip():
                    parsed_action.append(pre_action)
                if match_action.strip():
                    eof = first_match.group(3).strip()
                    if not match_action.split("\n")[0].strip().endswith(f"<< '{eof}'"):
                        guarded_command = match_action[first_match.start():]
                        first_line = guarded_command.split("\n")[0]
                        guarded_command = guarded_command.replace(
                            first_line, first_line + f" << '{eof}'", 1
                        )
                        parsed_action.append(guarded_command)
                    else:
                        parsed_action.append(match_action)
            else:
                parsed_action.append(rem_action)
                rem_action = ""
        return "\n".join(parsed_action)

    def split_actions(self, action: str, pattern_type="subroutine") -> List[Dict[str, Any]]:
        """Split an action into a list of actions in a greedy manner, each of which is a subroutine call or a single command."""
        parsed_action = list()
        rem_action = action
        while rem_action.strip():
            first_match = self._get_first_match(rem_action, pattern_type)
            if first_match:
                pre_action = rem_action[: first_match.start()]
                match_action = rem_action[first_match.start(): first_match.end()]
                rem_action = rem_action[first_match.end():]
                if pre_action.strip():
                    parsed_action.append(
                        {"agent": self.name, "action": pre_action, "cmd_name": None}
                    )
                if match_action.strip():
                    if match_action.split()[0] == self.config.submit_command:
                        parsed_action.append(
                            {
                                "agent": self.name,
                                "action": match_action,
                                "cmd_name": first_match.group(1),
                            }
                        )  # submit command is not a subroutine
                    else:
                        parsed_action.append(
                            {
                                "agent": first_match.group(1),
                                "args": first_match.group(2),
                                "action": match_action,
                                "cmd_name": first_match.group(1),
                            }
                        )
            else:
                parsed_action.append(
                    {"agent": self.name, "action": rem_action, "cmd_name": None}
                )
                rem_action = ""
        return parsed_action

    def _parse_command_patterns(self):
        assert self.config is not None  # mypy
        self.command_patterns = dict()
        for command in self.config._commands:
            if command.end_name is not None:
                pat = re.compile(
                    rf"^\s*({command.name})\s*(.*?)^({command.end_name})\s*$",
                    re.DOTALL | re.MULTILINE,
                )
                self.command_patterns[command.name] = pat
            else:
                pat = re.compile(rf"^\s*({command.name})\s*(.*?)$", re.MULTILINE)
                self.command_patterns[command.name] = pat
        self.subroutine_patterns = dict()
        for _, subroutine in self.config._subroutines.items():
            if subroutine.end_name is None:
                pat = re.compile(rf"^\s*({subroutine.name})\s*(.*?)$", re.MULTILINE)
                self.subroutine_patterns[subroutine.name,] = pat
            else:
                pat = re.compile(
                    rf"^\s*({subroutine.name})\s*(.*?)^({subroutine.end_name})\s*$",
                    re.DOTALL | re.MULTILINE,
                )
                self.subroutine_patterns[subroutine.name] = pat
        if hasattr(self.config, "submit_command_end_name"):
            submit_pat = re.compile(
                rf"^\s*({self.config.submit_command})\s*(.*?)^({self.config.submit_command_end_name})\s*$",
                re.DOTALL | re.MULTILINE,
            )
        else:
            submit_pat = re.compile(
                rf"^\s*({self.config.submit_command})(\s*)$", re.MULTILINE
            )  # group 2 is nothing
        self.subroutine_patterns[self.config.submit_command] = submit_pat
        self.command_patterns[self.config.submit_command] = submit_pat

    def read_history(self):
        history_content = ''
        for h in self.history:
            if 'is_demo' in h:
                continue
            role = h['role']
            if role == 'assistant':
                role = self.name
                action = h['action']
                thought = h['thought']
                history_content += f'{role} thinks: {thought}\n\n{role} action: {action}\n\n'
                continue
            elif role == 'user':
                role = 'Computer response'
            content = h['content']
            history_content += f'{role}: {content}\n\n'
        return history_content

    def guard_against_repeating_action_loop(self, thought, action, output, state_vars):
        solutions = {
            'edit': ' I should edit or rewrite a bigger section of the code at once',
            'goto': ' I just used goto command to this line, I should now execute some other command'
        }
        error_feedbacks = [
            'Your proposed edit has introduced new syntax error(s). Please understand the fixes and retry your edit commmand.',
            'Error:',
        ]

        is_goto_action = action.startswith('goto')
        open_file = state_vars['open_file']
        failed = False
        last_goto_line = None
        for i in range(1, min(10, len(self.history))):
            if self.name == self.history[-i]['agent']:
                if 'user' == self.history[-i]['role']:
                    if any(self.history[-i]['content'].startswith(it) for it in error_feedbacks):
                        failed = True
                elif 'assistant' == self.history[-i]['role']:
                    if is_goto_action and self.history[-i]['action'].startswith('goto'):
                        try:
                            line = int(action.split()[-1])
                            if last_goto_line is not None and abs(last_goto_line - line) < 60:
                                return f"When scanning a file I should use the cat command to read the whole file at a time.", 'pwd && ls -la', output
                        except ValueError:
                            line = None
                        last_goto_line = line
                    elif action == self.history[-i]['action']:
                        if failed and (action != 'edit' or self.history[-i]['open_file'] == open_file):
                            act = action.split()[0]
                            solution = solutions[act] if act in solutions else ''
                            command_outcome = ' that previously failed' if failed else ''
                            return f"I am again trying to execute the command {act}{command_outcome}. I should try a different approach.{solution}", 'pwd && ls -la', output
                        last_goto_line = None
                    else:
                        last_goto_line = None
                    failed = False
                else:
                    failed = False
                    last_goto_line = None
        return thought, action, output

    def forward(
            self, observation: str, available_actions: list[str], state: str, max_steps_reached: bool = False
    ) -> Tuple[str, str, str]:
        state_vars = json.loads(state)
        if max_steps_reached:
            thought, action, output = 'Pause due to max steps executed', 'exit_steps', 'Pause due to max steps executed'
        else:
            thought, action, output = self.forward_with_error_check(observation, state_vars)
            thought, action, output = self.guard_against_repeating_action_loop(thought, action, output, state_vars)

        self.history.append(
            {
                "role": "assistant",
                "content": output,
                "thought": thought,
                "action": action,
                "agent": self.name,
                "open_file": state_vars['open_file']
            }
        )

        logger.info(f"💭 THOUGHT ({self.name})\n{thought}")
        logger.info(f"🎬 ACTION ({self.name})\n{action}")

        return thought, action, output

    def forward_model(self, observation: str, state_vars: dict) -> str:
        """Query the model with the current state and observation with the appropriate template.

        Returns the model output."""
        assert self.config is not None  # mypy

        templates: List[str] = []
        # Determine observation template based on what prior observation was
        if self.history[-1]["role"] == "system" or self.history[-1].get(
                "is_demo", False
        ):
            # Show instance template if prev. obs. was initial system message
            templates = [self.config.instance_template]
            if self.config.strategy_template is not None:
                templates.append(self.config.strategy_template)
        elif observation is None or observation.strip() == "":
            # Show no output template if observation content was empty
            templates = [self.config.next_step_no_output_template]
        else:
            # Show standard output template if there is observation content
            templates = [self.config.next_step_template]

        # Populate selected template(s) with information (e.g., issue, arguments, state)
        messages = []
        for template in templates:
            messages.append(
                template.format(
                    **self.instance_args,
                    **self.system_args,
                    **state_vars,
                    observation=(observation if observation is not None else ""),
                )
            )

        message = "\n".join(messages)

        logger.info(f"🤖 MODEL INPUT\n{message}")
        self.history.append({"role": "user", "content": message, "agent": self.name})

        return self.model.query(self.local_history)

    def retry_after_format_fail(self, output):
        """Ask the model to correct (without committing to persistent history) after a malformatted model output"""
        format_error_template = self.config.format_error_template

        logger.warning(f"MALFORMED OUTPUT\n{output}")
        logger.warning(f"FORMAT ERROR\n{format_error_template}")

        temp_history = self.local_history + [
            {"role": "assistant", "content": output, "agent": self.name},
            {"role": "user", "content": format_error_template, "agent": self.name},
        ]
        return self.model.query(temp_history)

    def retry_after_blocklist_fail(self, output, action):
        """Ask the model to correct (without committing to persistent history) after a disallowed command"""
        name = action.strip().split()[0]
        blocklist_error_message = self.config.blocklist_error_template.format(name=name)

        logger.warning(f"BLOCKLISTED OUTPUT\n{output}")
        logger.warning(f"BLOCKLIST ERROR\n{blocklist_error_message}")

        temp_history = self.local_history + [
            {"role": "assistant", "content": output, "agent": self.name},
            {"role": "user", "content": blocklist_error_message, "agent": self.name},
        ]
        return self.model.query(temp_history)

    def should_block_action(self, action):
        """Check if the command should be blocked."""
        names = action.strip().split()
        if len(names) == 0:
            return False
        name = names[0]
        if name in self.config.blocklist:
            return True
        if name in self.config.blocklist_standalone and name == action.strip():
            return True
        return False

    def check_format_and_requery(
            self,
            output: str,
    ) -> Tuple[str, str, str]:
        """Query the model with the current state and observation with the appropriate template.

        Try to parse the output into a thought and action. Retry if the output is malformatted or the action is blocked.

        Returns the thought, action, and raw model output.
        """
        # Condition for handling outputs with no thought (just action)
        if self.model.args.model_name == "human":
            return "", output, output
        elif self.model.args.model_name == "human_thought":
            thought, action = ParseFunction.get("ThoughtActionParser")(
                output,
                self.config._commands + self.config.subroutine_types,
                strict=False,
            )
            return thought, action, output

        format_fails = blocklist_fails = 0

        while format_fails + blocklist_fails <= 2:
            try:
                thought, action = self.config.parse_function(
                    output,
                    self.config._commands + self.config.subroutine_types,
                    strict=False,
                )
            except KeyboardInterrupt:
                raise
            except FormatError:
                format_fails += 1
                output = self.retry_after_format_fail(output)
                continue
            if self.should_block_action(action):
                blocklist_fails += 1
                output = self.retry_after_blocklist_fail(output, action)
            else:
                return thought, action, output
        logger.warning(f"Malformat limit reached: \n{output}")
        return "Exit due to format error", "exit_format", output

    def forward_with_error_check(
            self, observation: str, state: str
    ) -> Tuple[str, str, str]:
        """Wrapper around `self.forward_model` that handles errors and retries
        due to format errors or blocked actions. 
        """
        try:
            output = self.forward_model(observation, state)
        except KeyboardInterrupt:
            raise
        except RuntimeError as e:
            logger.warning(f"Runtime error: {e}")
            return (
                f"Exit due to runtime error: {e}",
                "exit_error",
                f"exit due to runtime error: {e}",
            )
        except ContextWindowExceededError:
            logger.warning(f"Context window exceeded")
            return "Exit due to context window", "exit_context", "Exit due to context window"
        except CostLimitExceededError:
            logger.warning(f"Cost limit exceeded")
            return "Exit due to cost limit", "exit_cost", "Exit due to cost limit"
        except RetryError as e:
            logger.warning(f"Retry error: {e}")
            return (
                f"Exit due to retry error: {e}",
                "exit_api",
                f"exit due to retry error: {e}",
            )
        return self.check_format_and_requery(output)

    def init_environment_vars(self, env):
        self.set_environment_vars(env, self.config.env_variables)

    def set_environment_vars(self, env, env_variables):
        assert self.config is not None  # mypy
        commands_to_execute = (
                [self.config.state_command.code]
                +
                # [code for code in self.config.util_functions] +
                # [command.code for command in self.config._commands] +
                [f"{k}={v}" for k, v in env_variables.items()]
        )
        commands = "\n".join(commands_to_execute)
        try:
            output = env.communicate(commands)
            if env.returncode != 0:
                raise RuntimeError(
                    f"Nonzero return code: {env.returncode}\nOutput: {output}"
                )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.warning("Failed to set environment variables")
            raise e
        command_files = list()
        for file in self.config.command_files:
            datum = dict()
            contents = open(file, "r").read()
            datum["contents"] = contents
            filename = Path(file).name
            if not contents.strip().startswith("#!"):
                if filename.endswith(".sh"):
                    # files are sourced, so they are not executable
                    datum["name"] = Path(file).name
                    datum["type"] = "source_file"
                elif filename.startswith("_"):
                    # files are sourced, so they are not executable
                    datum["name"] = Path(file).name
                    datum["type"] = "utility"
                else:
                    raise ValueError(
                        (
                            f"Non-shell script file {file} does not start with shebang.\n"
                            "Either add a shebang (#!) or change the file extension to .sh if you want to source it.\n"
                            "You can override this behavior by adding an underscore to the file name (e.g. _utils.py)."
                        )
                    )
            else:
                # scripts are made executable
                datum["name"] = Path(file).name.rsplit(".", 1)[0]
                datum["type"] = "script"
            command_files.append(datum)
        env.add_commands(command_files)

    def get_environment_vars(self, env):
        assert self.config is not None  # mypy
        env_vars = dict()
        for var in self.config.env_variables:
            env_vars[var] = env.communicate(f"echo ${var}").strip()
        return env_vars

    def call_subroutine(self, agent_name, sub_action, env):
        assert self.config is not None  # mypy
        env_vars = self.get_environment_vars(env)
        cwd = env.communicate("pwd -P").strip()
        init_observation = self.config._subroutines[agent_name].init_observation
        if init_observation is not None:
            obs, _, _, _ = env.step(init_observation.format(args=sub_action["args"]))
        else:
            obs = None
        if env.returncode != 0:
            self.history.append({"role": "user", "content": obs, "agent": agent_name})
            raise RuntimeError(
                f"Nonzero return code: {env.returncode} for init_observation in {agent_name}.\n{obs}"
            )
        return_type = self.config._subroutines[agent_name].return_type
        sub_agent = Agent(agent_name, self.config._subroutines[agent_name].agent_args)
        sub_agent_output = sub_agent.run(
            {"issue": sub_action["args"]},
            env,
            observation=obs,
            return_type=return_type,
            init_model_stats=self.model.stats,
        )
        self.history += sub_agent.history
        self.set_environment_vars(env, env_vars)
        env.communicate(f"cd {cwd}")
        self.model.stats.replace(sub_agent.model.stats)
        return sub_agent_output

    def mc_setup(self,
                 setup_args: Dict[str, Any],
                 env: SWEEnv,
                 init_model_stats: Optional[APIStats] = None,
                 ):
        assert env.container_obj is not None
        assert self.config is not None  # mypy

        if env.container_obj.id != self.last_container_id:
            logger.info(
                f"Initializing agent settings for container {env.container_obj.id}"
            )
            self.init_environment_vars(env)
            self.last_container_id = env.container_obj.id
        # Re-initialize primary
        self.setup(setup_args, init_model_stats)

        # Init action/observation loop
        self.mc_trajectory = []
        self.mc_info = {}

    def receive_message(self, sender: str, msg: str):
        self.history.append({'role': 'user', 'content': f'{sender} said: {msg}', 'agent': self.name})

    def run_continue(self,
                     env: SWEEnv,
                     observation: Optional[str] = None,
                     traj_dir: Optional[Path] = None,
                     return_type: Optional[str] = "info",
                     max_steps: Optional[int] = None,
                     ):
        assert env.container_obj is not None
        assert self.config is not None  # mypy

        if env.container_obj.id != self.last_container_id:
            logger.info(
                f"Initializing agent settings for container {env.container_obj.id}"
            )
            self.init_environment_vars(env)
            self.last_container_id = env.container_obj.id

        trajectory = self.mc_trajectory
        done = False
        step = 0

        while not done:
            if env.repo_path:
                rm = RepoMap(root=env.repo_path)
                repo_map = rm.get_repo_map()
            else:
                repo_map = "n/a"
            self.system_args["repo_map"] = repo_map

            state = env.communicate(self.state_command) if self.state_command else None

            thought, action, output = self.forward(
                observation, env.get_available_actions(), state, max_steps and max_steps <= step
            )
            observations = list()
            run_action = self._guard_multiline_input(action)
            for sub_action in self.split_actions(run_action):
                if (
                        sub_action["agent"] == self.name
                        or sub_action["cmd_name"] == self.config.submit_command
                ):
                    obs, _, done, info = env.step(sub_action["action"])
                    self.mc_info = info
                    observations.append(obs)
                    if sub_action["cmd_name"] == self.config.submit_command:
                        done = True
                    if done:
                        break
                else:
                    agent_name = sub_action["agent"]
                    sub_agent_output = self.call_subroutine(agent_name, sub_action, env)
                    observations.append(sub_agent_output)

            observation = "\n".join([obs for obs in observations if obs is not None])

            trajectory.append(
                {
                    "action": action,
                    "observation": observation,
                    "response": output,
                    "state": state,
                    "thought": thought,
                }
            )
            self.mc_info["model_stats"] = self.model.stats.to_dict()
            if traj_dir:
                self.save_trajectory(trajectory, traj_dir, env, self.mc_info)
            step += 1
        if return_type == "info":
            return self.mc_info
        if return_type == "info_trajectory":
            return self.mc_info, trajectory
        return trajectory[-1][return_type]

    def run(
            self,
            setup_args: Dict[str, Any],
            env: SWEEnv,
            observation: Optional[str] = None,
            traj_dir: Optional[Path] = None,
            return_type: Optional[str] = "info",
            init_model_stats: Optional[APIStats] = None,
    ):
        """
        Run the agent on an environment.
        Return the final value of the specified return type.
        """
        done = False
        assert env.container_obj is not None
        assert self.config is not None  # mypy

        if env.container_obj.id != self.last_container_id:
            logger.info(
                f"Initializing agent settings for container {env.container_obj.id}"
            )
            self.init_environment_vars(env)
            self.last_container_id = env.container_obj.id
        # Re-initialize primary
        self.setup(setup_args, init_model_stats)

        # Run action/observation loop
        trajectory = []
        self.mc_info = {}
        self.mc_trajectory = trajectory
        while not done:
            if env.repo_path:
                rm = RepoMap(root=env.repo_path)
                repo_map = rm.get_repo_map()
            else:
                repo_map = "n/a"
            self.system_args["repo_map"] = repo_map

            state = env.communicate(self.state_command) if self.state_command else None
            thought, action, output = self.forward(
                observation, env.get_available_actions(), state
            )
            observations = list()
            run_action = self._guard_multiline_input(action)
            for sub_action in self.split_actions(run_action):
                if (
                        sub_action["agent"] == self.name
                        or sub_action["cmd_name"] == self.config.submit_command
                ):
                    obs, _, done, info = env.step(sub_action["action"])
                    self.mc_info = info
                    observations.append(obs)
                    if sub_action["cmd_name"] == self.config.submit_command:
                        done = True
                    if done:
                        break
                else:
                    agent_name = sub_action["agent"]
                    sub_agent_output = self.call_subroutine(agent_name, sub_action, env)
                    observations.append(sub_agent_output)

            observation = "\n".join([obs for obs in observations if obs is not None])

            trajectory.append(
                {
                    "action": action,
                    "observation": observation,
                    "response": output,
                    "state": state,
                    "thought": thought,
                }
            )
            self.mc_info["model_stats"] = self.model.stats.to_dict()
            if traj_dir:
                self.save_trajectory(trajectory, traj_dir, env, self.mc_info)
        if return_type == "info":
            return self.mc_info
        if return_type == "info_trajectory":
            return self.mc_info, trajectory
        return trajectory[-1][return_type]


@tool
def engineer(instructions: str):
    """Let the engineer continue work on the project with the given instructions.
    Call when the work presented by the engineer does not meet the requirements of the project or is not
    tested enough."""
    finish_tool(STATE_WRITE, instructions)


@tool
def manual_test(details: str, flows_to_test: str):
    """Instruct the engineer to manually test the software.
    Call when the work presented by the engineer requires further manual testing to ensure maximum
    production user satisfaction"""
    finish_tool(STATE_WRITE,
                f'The manager has requested you to manually test the software. {details}\n\nHere are the flows that should be tested:\n{flows_to_test}')


@tool
def write_automatic_tests(details: str, test_cases: str):
    """Request the engineer to write the automated tests."""
    finish_tool(STATE_WRITE,
                f'The manager has requested you to write automatic tests:\n{details}\n\nRequired test cases:\n{test_cases}')


@tool
def submit():
    """Submit the software product to production. Call only when you think the engineer has finished work
    meeting the requirements presented in the issue AND has sufficiently tested the software via running
    automated tests or manual testing. This can ONLY be called in case the tests have been implemented, run and PASSED"""
    finish_tool(STATE_DONE, None)


@tool
def forget_messages(search_keywords: list[str], delete_count: int):
    """
    Makes the engineer forget messages that contain a match to the `search_keyword` if there are exatly total
    `delete_count` matches in chat history. The corresponding user or assistant message is also deleted for each match.

    :param search_keywords: case-insensitive plain text search keyword
    :param delete_count: number of message pairs expected to forget
    """
    SWESwarm.the_swarm.forget_messages(search_keywords, delete_count)

@tool
def forget_message(search_keyword: str, delete_count: int = 1):
    """
    Makes the engineer forget messages that contain a match to the `search_keyword` if there are exatly total
    `delete_count` matches in chat history. The corresponding user or assistant message is also deleted for each match.

    :param search_keyword: case-insensitive plain text search keyword
    :param delete_count: number of message pairs expected to forget
    """
    SWESwarm.the_swarm.forget_messages([search_keyword], delete_count)


@tool
def noop():
    """No operation"""
    finish_tool(SWESwarm.the_swarm.state, SWESwarm.the_swarm.phase_parameters)


STATE_WRITE = 'write'
STATE_MANAGE = 'manage'
STATE_DONE = 'done'


def finish_tool(_state, _phase_parameters):
    if SWESwarm.the_swarm is None:
        raise RuntimeError('State error')
    SWESwarm.the_swarm.state = _state
    SWESwarm.the_swarm.phase_parameters = _phase_parameters


import config

cfg = config.Config(os.path.join(os.getcwd(), "keys.cfg"))
vertexai.init(project=cfg.VERTEX_PROJECT, location=cfg.VERTEX_LOCATION)

llm = ChatVertexAI(model="gemini-1.5-pro-preview-0409")

manager_tools = [engineer, manual_test, write_automatic_tests, submit]
llm_mgr = llm.bind_tools(manager_tools)
manager_tools_map = {x.name: x for x in manager_tools}

memory_manager_tools = [forget_messages, forget_message]
llm_mmgr = llm.bind_tools(manager_tools)
memory_manager_tools_map = {x.name: x for x in memory_manager_tools}


class LCAgent:
    def __init__(self, llm: Runnable, system: str):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{input}"),
            ]
        )
        self.chain = prompt | llm

    def invoke(
            self,
            input: Input,
            config: Optional[RunnableConfig] = None,
            **kwargs: Optional[Any],
    ) -> Output:
        for i in range(10):
            try:
                return self.chain.invoke(input, config, **kwargs)
            except ResourceExhausted:
                pass
            except RateLimitError:
                pass
            except ValueError as e:
                if "Multiple content parts are not supported" in str(e):
                    continue
                raise e
            except IndexError as e:
                if "list index out of range" in str(e):
                    continue
                raise e
            except InternalServerError:
                logger.warning('Internal server error occurred in LLM provider, waiting 15 seconds')
                logger.debug(input)
                sleep(15)
                continue
            time = int(60 * 2 ** i)
            logger.warning(f'Rate limit exceeded, waiting {time} seconds')
            sleep(time)


import rich.console
import rich.markdown
import rich.panel
import rich.markdown


def _print_patch_message(patch_output_file: Path):
    console = rich.console.Console()
    msg = [
        "SWE-agent has produced a patch that it believes will solve the issue you submitted!",
        "Use the code snippet below to inspect or apply it!"
    ]
    panel = rich.panel.Panel.fit(
        "\n".join(msg),
        title="🎉 Submission successful 🎉",
    )
    console.print(panel)
    content = [
        "```bash",
        f"# The patch has been saved to your local filesystem at:",
        f"PATCH_FILE_PATH='{patch_output_file.resolve()}'",
        "# Inspect it:",
        "cat \"${PATCH_FILE_PATH}\"",
        "# Apply it to a local repository:",
        f"cd <your local repo root>",
        "git apply \"${PATCH_FILE_PATH}\"",
        "```",
    ]
    console.print(rich.markdown.Markdown("\n".join(content)))


def save_predictions(traj_dir: Path, instance_id: str, info):
    output_file = traj_dir / "all_preds.jsonl"
    model_patch = info["submission"] if "submission" in info else None
    datum = {
        KEY_MODEL: Path(traj_dir).name,
        KEY_INSTANCE_ID: instance_id,
        KEY_PREDICTION: model_patch,
    }
    with open(output_file, "a+") as fp:
        print(json.dumps(datum), file=fp, flush=True)
    logger.info(f"Saved predictions to {output_file}")


def save_patch(traj_dir: Path, instance_id: str, info) -> Optional[Path]:
    """Create patch files that can be applied with `git am`.

    Returns:
        The path to the patch file, if it was saved. Otherwise, returns None.
    """
    patch_output_dir = traj_dir / "patches"
    patch_output_dir.mkdir(exist_ok=True, parents=True)
    patch_output_file = patch_output_dir / f"{instance_id}.patch"
    if not info.get("submission"):
        logger.info("No patch to save.")
        return
    model_patch = info["submission"]
    patch_output_file.write_text(model_patch)
    _print_patch_message(patch_output_file)
    return patch_output_file


def apply_patch(local_dir: Path, patch_file: Path) -> None:
    """Apply a patch to a local directory."""
    assert local_dir.is_dir()
    assert patch_file.exists()
    # The resolve() is important, because we're gonna run the cmd
    # somewhere else
    cmd = ["git", "apply", "--whitespace=fix", str(patch_file.resolve())]
    cmd_status = ["git", "status"]
    cmd_add = ["git", "add", "."]
    cmd_commit = ["git", "commit", "-m" f'SWE-Agent applied patch {str(patch_file.resolve())}']
    try:
        subprocess.run(cmd, cwd=local_dir, check=True)
        subprocess.run(cmd_status, cwd=local_dir, check=True)
        subprocess.run(cmd_add, cwd=local_dir, check=True)
        subprocess.run(cmd_commit, cwd=local_dir, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to apply patch {patch_file} to {local_dir}: {e}")
        return
    logger.info(f"Applied patch {patch_file} to {local_dir}")


class SWESwarm:
    the_swarm = None

    def __init__(self, args):
        self.args = args
        self.engineer_agent = Agent("engineer", args.engineer_agent)
        self.manager_agent = LCAgent(
            llm_mgr,
            """You are a manager at a software development team with one engineer. You are responsible for the 
            quality of the engineer's work and organizing their work. As a response just call one of the functions 
            provided to indicate how the engineer should proceed or submit if you are certain based on the testing 
            output that work is done. If the agent exits due to any error, it is time to check on it's progress and 
            let it resume if on right track or instruct to take a better approach. You never answer to requests from 
            the engineer, you provide instructions to make the engineer complete the task efficiently. It is important 
            to avoid the engineer getting stuck and wasting time as their working hours are costly. Remember, the 
            best way to ensure reliability of the software is by verifying automated test results and coverage and 
            their match to product requirements (the ISSUE). Answer as if you are talking directly to the engineer"""
        )
        self.memory_manager_agent = LCAgent(
            llm_mmgr,
            """You are the memory manager agent of a software engineer working in a software development team. 
            You are responsible for the quality of the engineer's mental space and ensuring that they can focus on the 
            most important issues the they work on, without deleting important information from their memory. 
            As a response, call the forget_message or forget_messages functions with search_keyword(s) to select 
            message(s) to be deleted from the engineer's memory. In order to maximize the engineer's performance 
            you should delete memories 
            1) where the agent successfully solved local bugs like syntax errors, 
            2) where the agent prepared the code execution environment like installing missing software, 
            3) that introduce duplicate information - there is no need to repeat unless it is very important
            4) where the agent is repeating similar actions, especially if with errors
            5) messages from the manager agent that are old and have become irrelevant to the engineer's situation
            6) other messages that are not important for the engineer to remember
            
            You should not delete messages that contain important information about the project, the issue like the 
            first 4 messages and the last 6 messages (most recent memories)
            
            Use each keyword to select a unique sentence in the history. The delete count should usually equal to the 
            number of keywords. For example if there is a message 
            `The code seems to have the basic functionality for controlling the switch. However, there are some areas that need attention: ...`
            use keyword `basic functionality for controlling the switch. However` to match this sentence and provide 
            delete_count=1 to ensure the correct message is deleted."""
        )
        self.state = STATE_WRITE
        self.phase_parameters = None

    def forget_messages(self, search_keywords: list[str], delete_count: int):
        self.engineer_agent.forget_messages(search_keywords, delete_count)

    def run(
            self,
            setup_args: Dict[str, Any],
            env: SWEEnv,
            observation: Optional[str] = None,
            traj_dir: Optional[Path] = None,
            return_type: Optional[str] = "info",
            init_model_stats: Optional[APIStats] = None,
            actions: Any = None,
            instance_id: Optional[str] = None,
    ):
        """
        Run the agent on an environment.
        Return the final value of the specified return type.
        """

        self.engineer_agent.mc_setup(setup_args, env, init_model_stats)

        latest_patch_path = None
        i = 1
        try:
            while self.state != STATE_DONE:

                if self.state == STATE_WRITE:
                    # self.engineer_agent.model.reset_stats(init_model_stats)
                    if self.phase_parameters is not None:
                        self.engineer_agent.receive_message('manager', self.phase_parameters)
                    t_dir = Path(str(traj_dir.absolute()) + f'iteration{i}')
                    t_dir.mkdir(exist_ok=True)
                    info, trajectory = self.engineer_agent.run_continue(env, observation, t_dir, return_type, 10)
                    save_predictions(t_dir, instance_id, info)
                    latest_patch_path = save_patch(t_dir, instance_id, info)
                    self.state = STATE_MANAGE
                elif self.state == STATE_MANAGE:
                    progress = self.engineer_agent.read_history()
                    if len(self.engineer_agent.history) > 10:
                        out = self.memory_manager_agent.invoke(
                            f'I am the software engineer, here is my message history memory: {progress}'
                        )
                        try:
                            for x in out.tool_calls:
                                memory_manager_tools_map[x['name']](tool_input=x['args'])
                        except Exception as e:
                            logger.error('Tool call failed.')
                            logger.error(e)

                    progress_mgr = progress.replace('manager', 'manager (you)')
                    out = self.manager_agent.invoke(
                        f'I am the software engineer, here is my progress so far: {progress_mgr}'
                    )
                    try:
                        for x in out.tool_calls:
                            manager_tools_map[x['name']](tool_input=x['args'])
                        if len(out.tool_calls) == 0:
                            engineer(tool_input={'instructions': out.content})
                    except Exception as e:
                        logger.error('Tool call failed.')
                        logger.error(e)
                        self.state = STATE_DONE

                i += 1
        finally:
            if actions.apply_patch_locally and latest_patch_path is not None and env.record["repo_type"] == "local":
                apply_patch(Path(env.repo_path), patch_file=latest_patch_path)
