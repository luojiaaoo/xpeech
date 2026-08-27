// Copyright (c) 2026 Xpeech contributors.
// SPDX-License-Identifier: MIT

// This wrapper keeps the upstream lark-cli command tree intact while replacing
// its credential source with the provider generated from Xpeech's conf.toml.
package main

import (
	"os"

	"github.com/larksuite/cli/cmd"

	_ "github.com/larksuite/cli/mycred"
)

func main() {
	os.Exit(cmd.Execute())
}
