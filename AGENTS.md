# AGENTS.md

## Project overview

This project is Delta Exit Assistant / 三角洲下机助手.

It is a Windows desktop helper tool for reminding the user to stop playing after a match ends.
The main purpose is not game automation or cheating. It should only detect UI states locally and remind the user through tray notification, popup, or sound.

## Core user value

The user often forgets to stop after the match settlement screen.
The app should detect the settlement/end state with minimal false positives, then remind the user to take a break or exit the game.

## Important behavior constraints

- Do not implement cheating, memory reading, packet sniffing, input automation, aiming assistance, recoil control, or anything that modifies the game process.
- Prefer screen/image recognition, window detection, template matching, and local UI logic.
- Keep changes minimal and reversible.
- Do not perform large refactors unless explicitly requested.
- Preserve existing user-facing Chinese text unless the task asks to change it.
- Maintain Windows compatibility.
- If packaging/build scripts already exist, do not replace them casually.

## Expected workflow

Before editing code:
1. Read README, source files, config files, build scripts, and existing docs.
2. Summarize the current architecture.
3. Identify the entry point, main modules, dependencies, and data flow.
4. Explain what files are likely affected by the requested change.
5. Ask for confirmation before large structural changes.

When editing:
1. Make the smallest correct change.
2. Explain the diff.
3. Run available tests or basic smoke checks if possible.
4. If tests do not exist, explain how the change was manually verified or how the user can verify it.

## Coding style

- Prefer simple, explicit Python code.
- Avoid clever abstractions.
- Keep UI logic, detection logic, config logic, and packaging logic separated where possible.
- Add comments only where the logic is non-obvious.
