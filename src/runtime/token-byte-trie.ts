export interface TokenByteSource {
  readonly vocabularySize: number;
  tokenBytes(tokenId: number): Uint8Array | null;
}

export interface TokenByteTrieNode {
  readonly children: ReadonlyMap<number, TokenByteTrieNode>;
  readonly tokenIds: readonly number[];
}

interface MutableTokenByteTrieNode {
  children: Map<number, MutableTokenByteTrieNode>;
  tokenIds: number[];
}

export class TokenByteTrie {
  readonly root: TokenByteTrieNode;
  readonly tokenCount: number;

  constructor(source: TokenByteSource) {
    const root: MutableTokenByteTrieNode = { children: new Map(), tokenIds: [] };
    let tokenCount = 0;
    for (let tokenId = 0; tokenId < source.vocabularySize; tokenId += 1) {
      const bytes = source.tokenBytes(tokenId);
      if (!bytes || bytes.length === 0) continue;
      let node = root;
      for (const byte of bytes) {
        let child = node.children.get(byte);
        if (!child) {
          child = { children: new Map(), tokenIds: [] };
          node.children.set(byte, child);
        }
        node = child;
      }
      node.tokenIds.push(tokenId);
      tokenCount += 1;
    }
    this.root = root;
    this.tokenCount = tokenCount;
  }
}
