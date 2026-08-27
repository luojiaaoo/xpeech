package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestRefreshUserTokenRotatesAndValidatesResponse(t *testing.T) {
	cached := expiredCachedUserToken()
	server := refreshTestServer(t, http.StatusOK, `{
		"access_token": "new-access",
		"refresh_token": "new-refresh",
		"expires_in": 7200,
		"refresh_token_expires_in": 604800,
		"scope": "offline_access contact:user.base:readonly"
	}`)
	defer server.Close()
	useRefreshEndpoint(t, server.URL)

	refreshed, err := refreshUserToken(context.Background(), cached)
	if err != nil {
		t.Fatalf("refreshUserToken() error = %v", err)
	}
	if refreshed.AccessToken != "new-access" || refreshed.RefreshToken != "new-refresh" {
		t.Fatalf("refreshUserToken() token = %#v", refreshed)
	}
	if refreshed.RefreshToken == cached.RefreshToken {
		t.Fatal("refresh token was not rotated")
	}
	if !usableAccessToken(refreshed) || !reusableRefreshToken(refreshed) {
		t.Fatalf("refreshed token is not reusable: %#v", refreshed)
	}
}

func TestRefreshUserTokenRejectsMissingOrUnrotatedRefreshToken(t *testing.T) {
	tests := map[string]string{
		"missing": `{
			"access_token": "new-access",
			"expires_in": 7200,
			"refresh_token_expires_in": 604800
		}`,
		"unrotated": `{
			"access_token": "new-access",
			"refresh_token": "old-refresh",
			"expires_in": 7200,
			"refresh_token_expires_in": 604800
		}`,
	}
	for name, body := range tests {
		t.Run(name, func(t *testing.T) {
			server := refreshTestServer(t, http.StatusOK, body)
			defer server.Close()
			useRefreshEndpoint(t, server.URL)

			_, err := refreshUserToken(context.Background(), expiredCachedUserToken())
			if err == nil || !refreshFailureRequiresAuthorization(err) {
				t.Fatalf("refreshUserToken() error = %v, want reauthorization error", err)
			}
		})
	}
}

func TestRefreshFailureClassification(t *testing.T) {
	terminalCodes := []int{
		feishuRefreshInvalid,
		feishuRefreshExpired,
		feishuRefreshRevoked,
		feishuRefreshAlreadyUsed,
	}
	for _, code := range terminalCodes {
		t.Run(fmt.Sprintf("code_%d", code), func(t *testing.T) {
			body := fmt.Sprintf(`{"code":%d,"msg":"refresh token is unusable"}`, code)
			server := refreshTestServer(t, http.StatusBadRequest, body)
			defer server.Close()
			useRefreshEndpoint(t, server.URL)

			_, err := refreshUserToken(context.Background(), expiredCachedUserToken())
			if err == nil || !refreshFailureRequiresAuthorization(err) {
				t.Fatalf("refreshUserToken() error = %v, want reauthorization error", err)
			}
		})
	}

	t.Run("invalid_grant", func(t *testing.T) {
		server := refreshTestServer(t, http.StatusBadRequest,
			`{"error":"invalid_grant","error_description":"refresh token expired"}`)
		defer server.Close()
		useRefreshEndpoint(t, server.URL)

		_, err := refreshUserToken(context.Background(), expiredCachedUserToken())
		if err == nil || !refreshFailureRequiresAuthorization(err) {
			t.Fatalf("refreshUserToken() error = %v, want reauthorization error", err)
		}
	})

	t.Run("server_error", func(t *testing.T) {
		server := refreshTestServer(t, http.StatusServiceUnavailable,
			`{"code":50000,"msg":"temporarily unavailable"}`)
		defer server.Close()
		useRefreshEndpoint(t, server.URL)

		_, err := refreshUserToken(context.Background(), expiredCachedUserToken())
		if err == nil || refreshFailureRequiresAuthorization(err) {
			t.Fatalf("refreshUserToken() error = %v, want retryable error", err)
		}
	})
}

func TestShouldRefreshOnlyForExpiredAccessWithExistingScopes(t *testing.T) {
	desired := []string{"offline_access"}
	cached := expiredCachedUserToken()
	if !shouldRefreshUserToken(cached, desired) {
		t.Fatal("expired access token with reusable refresh token should refresh")
	}

	cached.ExpiresAt = time.Now().Add(time.Hour).Unix()
	if shouldRefreshUserToken(cached, desired) {
		t.Fatal("usable access token should not refresh")
	}

	cached.ExpiresAt = time.Now().Add(-time.Hour).Unix()
	if shouldRefreshUserToken(cached, []string{"offline_access", "docs:doc:readonly"}) {
		t.Fatal("refresh must not be used to request a missing scope")
	}

	cached.RefreshTokenExpiresAt = 0
	if reusableRefreshToken(cached) {
		t.Fatal("refresh token with unknown expiry must not be reused")
	}
}

func TestSaveUserTokenReliablyUsesPrivateAtomicCache(t *testing.T) {
	path := filepath.Join(t.TempDir(), "xpeech", "lark-cli-user-token.json")
	token := expiredCachedUserToken()
	token.AccessToken = "persisted-access"

	if err := saveUserTokenReliably(path, token); err != nil {
		t.Fatalf("saveUserTokenReliably() error = %v", err)
	}
	loaded, err := loadUserToken(path)
	if err != nil {
		t.Fatalf("loadUserToken() error = %v", err)
	}
	if !cachedUserTokensEqual(loaded, token) {
		t.Fatalf("loadUserToken() = %#v, want %#v", loaded, token)
	}
	assertFileMode(t, filepath.Dir(path), 0o700)
	assertFileMode(t, path, 0o600)
}

func expiredCachedUserToken() *cachedUserToken {
	now := time.Now()
	return &cachedUserToken{
		AppID:                 appID,
		AccessToken:           "old-access",
		RefreshToken:          "old-refresh",
		Scope:                 "offline_access contact:user.base:readonly",
		ExpiresAt:             now.Add(-time.Hour).Unix(),
		RefreshTokenExpiresAt: now.Add(time.Hour).Unix(),
	}
}

func refreshTestServer(t *testing.T, status int, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if err := request.ParseForm(); err != nil {
			t.Errorf("ParseForm() error = %v", err)
		}
		if request.Method != http.MethodPost || request.Form.Get("grant_type") != "refresh_token" {
			t.Errorf("unexpected refresh request: method=%s form=%v", request.Method, request.Form)
		}
		if request.Form.Get("client_id") != appID || request.Form.Get("client_secret") != appSecret {
			t.Error("refresh request did not contain the configured client credentials")
		}
		if request.Form.Get("refresh_token") != "old-refresh" {
			t.Errorf("refresh_token = %q, want old-refresh", request.Form.Get("refresh_token"))
		}
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(status)
		if _, err := writer.Write([]byte(strings.TrimSpace(body))); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
}

func useRefreshEndpoint(t *testing.T, endpoint string) {
	t.Helper()
	previous := oauthTokenEndpoint
	oauthTokenEndpoint = endpoint
	t.Cleanup(func() {
		oauthTokenEndpoint = previous
	})
}

func assertFileMode(t *testing.T, path string, expected os.FileMode) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	if actual := info.Mode().Perm(); actual != expected {
		t.Fatalf("mode of %s = %o, want %o", path, actual, expected)
	}
}
