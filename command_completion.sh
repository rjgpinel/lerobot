#!/bin/bash

_commands_sh_autocomplete() {
  local cur prev opts
  COMPREPLY=()
  cur="${COMP_WORDS[COMP_CWORD]}"
  opts="run_act_chicken run_dp_chicken run_act_cube set_static_cam_controls"

  COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
  return 0
}

complete -F _commands_sh_autocomplete ./commands.sh
