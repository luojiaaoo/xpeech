function userId(r) {
    // Internal requests pass user IDs in headers; external APIs ignore this header.
    const fallback = r.headersIn['X-User-Id'] || '';
    const bearer = (r.headersIn.Authorization || '').match(/^Bearer\s+(\S+)$/i);
    if (!bearer) return fallback;
    try {
        const claims = JSON.parse(Buffer.from(bearer[1].split('.')[1], 'base64url').toString());
        return typeof claims.sub === 'string' ? claims.sub : fallback;
    } catch (_) {
        return fallback;
    }
}

export default { userId };
