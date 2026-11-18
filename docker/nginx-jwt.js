function userId(r) {
    // User affinity comes only from JWT sub; verification stays in the backend.
    const bearer = (r.headersIn.Authorization || '').match(/^Bearer\s+(\S+)$/i);
    if (!bearer) return '';
    try {
        const claims = JSON.parse(Buffer.from(bearer[1].split('.')[1], 'base64url').toString());
        return typeof claims.sub === 'string' ? claims.sub : '';
    } catch (_) {
        return '';
    }
}

export default { userId };
