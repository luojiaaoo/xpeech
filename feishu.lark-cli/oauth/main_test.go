package main

import "testing"

func TestScopesFlagSplitsAndPreservesValues(t *testing.T) {
    var scopes scopesFlag
    if err := scopes.Set("offline_access, docs:doc:readonly"); err != nil { t.Fatal(err) }
    if len(scopes) != 2 || scopes[0] != "offline_access" || scopes[1] != "docs:doc:readonly" { t.Fatalf("scopes = %#v", scopes) }
}

func TestDomainsFlagSplitsAndPreservesValues(t *testing.T) {
    var domains domainsFlag
    if err := domains.Set("calendar, task"); err != nil { t.Fatal(err) }
    if len(domains) != 2 || domains[0] != "calendar" || domains[1] != "task" { t.Fatalf("domains = %#v", domains) }
}

func TestMergeScopesDeduplicates(t *testing.T) {
    got := mergeScopes([]string{"calendar:calendar:read", "docs:doc:readonly"}, []string{"docs:doc:readonly", "drive:drive:readonly"})
    want := []string{"calendar:calendar:read", "docs:doc:readonly", "drive:drive:readonly"}
    if len(got) != len(want) { t.Fatalf("mergeScopes = %#v, want %#v", got, want) }
    for i := range want { if got[i] != want[i] { t.Fatalf("mergeScopes = %#v, want %#v", got, want) } }
}

func TestScopesForDomainsIncludesNativeCalendarScopes(t *testing.T) {
    scopes, err := scopesForDomains([]string{"calendar"})
    if err != nil { t.Fatal(err) }
    if len(scopes) == 0 { t.Fatal("calendar domain returned no scopes") }
}

func TestLoginRequiresScopeOrDomain(t *testing.T) {
    for _, args := range [][]string{
        nil,
        {"login"},
        {"authorize"},
        {"login", "--scope", ""},
        {"login", "--scope", " , ; "},
    } {
        if got := run(args); got != 2 { t.Errorf("run(%q) = %d, want 2", args, got) }
    }
}
