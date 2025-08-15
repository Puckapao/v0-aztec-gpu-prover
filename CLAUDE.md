# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Primary Commands
- **Build entire repository**: `./bootstrap.sh` or `./bootstrap.sh fast` (uses S3 cache)
- **Build from scratch**: `./bootstrap.sh full` (no cache)
- **Run tests**: `./bootstrap.sh test`
- **Clean repository**: `./bootstrap.sh clean` (removes untracked files)
- **Format code**: `./bootstrap.sh format`
- **Lint code**: `./bootstrap.sh lint`

### Yarn Project Commands (from yarn-project/)
- **Build TypeScript**: `yarn build` or `tsc -b`
- **Watch mode**: `yarn build:dev` or `tsc -b -w`
- **Generate code**: `yarn generate`
- **Clean workspaces**: `yarn clean`

### L1 Contracts (from l1-contracts/)
- **Format Solidity**: `yarn format` (uses forge fmt)
- **Lint Solidity**: `yarn lint`
- **Run security analysis**: `yarn slither`

### Testing
- **Get test commands**: `./bootstrap.sh test_cmds` (lists all available tests)
- **Individual project tests**: `./*/scripts/run_test.sh` (each project has this script)
- **Run specific test**: Use the commands returned by `test_cmds`

## Architecture Overview

### Core Components
- **barretenberg/**: ZK prover backend and Aztec VM implementation in C++
- **l1-contracts/**: Ethereum rollup contracts in Solidity
- **noir-projects/**: Noir smart contracts and protocol circuits
- **yarn-project/**: TypeScript client libraries and backend services

### Key TypeScript Packages
- **aztec.js/**: Main client library for interacting with Aztec network
- **pxe/**: Private Execution Environment (PXE) for private state management
- **aztec-node/**: Full node implementation
- **sequencer-client/**: Transaction sequencing logic
- **prover-client/**: Proof generation coordination
- **cli/**: Command-line interface tools
- **end-to-end/**: Integration tests and examples

### Noir Components
- **aztec-nr/**: Noir framework for Aztec smart contracts
- **noir-contracts/**: Example contracts and protocol implementations
- **noir-protocol-circuits/**: Core protocol circuits

## Development Workflow

### Bootstrapping
The repository uses a sophisticated bootstrap system (CI3) that:
- Downloads artifacts from S3 cache when possible
- Builds dependencies in topological order
- Handles complex cross-language builds (C++, Rust, TypeScript, Noir)

### Monorepo Structure
- Yarn workspaces manage TypeScript packages
- Each component has its own `bootstrap.sh` script
- Tests can be run individually or in parallel across the entire repo
- CI uses a single large machine (128 vcpu) for fast parallel execution

### Package Management
- **Yarn 4.5.2** for TypeScript dependencies
- **Portal/file:** resolutions link local packages (bb.js, noir packages)
- Prettier and ESLint configured at workspace level

### Building Dependencies
Key build dependencies:
- Barretenberg must be built before TypeScript packages (provides bb.js)
- Noir repository is mirrored using git-subrepo
- Protocol circuits generate TypeScript types for client packages

## Testing Patterns
- Each project exposes tests via `./bootstrap.sh test_cmds`
- Tests are designed to run in parallel for maximum throughput
- Integration tests in `yarn-project/end-to-end/` serve as usage examples
- CI uses Redis cache to avoid re-running identical tests