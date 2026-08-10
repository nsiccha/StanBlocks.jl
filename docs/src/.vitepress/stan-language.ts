import type { LanguageRegistration } from '@shikijs/types'

// Stan is not bundled by Shiki 2.5. This compact TextMate grammar covers the
// language constructs emitted in these docs without sending code to a
// client-side highlighter or relabelling Stan as another language.
export const stanLanguage = {
  name: 'stan',
  displayName: 'Stan',
  scopeName: 'source.stan',
  patterns: [
    { include: '#comments' },
    { include: '#strings' },
    { include: '#blocks' },
    { include: '#types' },
    { include: '#constraints' },
    { include: '#keywords' },
    { include: '#constants' },
    { include: '#numbers' },
    { include: '#functions' },
    { include: '#operators' },
  ],
  repository: {
    comments: {
      patterns: [
        { begin: '/\\*', end: '\\*/', name: 'comment.block.stan' },
        { begin: '//', end: '$', name: 'comment.line.double-slash.stan' },
        { begin: '#', end: '$', name: 'comment.line.number-sign.stan' },
      ],
    },
    strings: {
      patterns: [
        {
          begin: '"',
          end: '"',
          name: 'string.quoted.double.stan',
          patterns: [
            { match: '\\\\.', name: 'constant.character.escape.stan' },
          ],
        },
      ],
    },
    blocks: {
      patterns: [
        {
          match: '\\b(functions|data|transformed\\s+data|parameters|transformed\\s+parameters|model|generated\\s+quantities)\\b(?=\\s*\\{)',
          name: 'entity.name.section.stan',
        },
      ],
    },
    types: {
      patterns: [
        {
          match: '\\b(int|real|complex|array|tuple|vector|row_vector|matrix|simplex|unit_vector|sum_to_zero_vector|sum_to_zero_matrix|ordered|positive_ordered|corr_matrix|cov_matrix|cholesky_factor_cov|cholesky_factor_corr|column_stochastic_matrix|row_stochastic_matrix|complex_vector|complex_row_vector|complex_matrix|void)\\b',
          name: 'storage.type.stan',
        },
      ],
    },
    constraints: {
      patterns: [
        {
          match: '\\b(lower|upper|offset|multiplier)\\b(?=\\s*=)',
          name: 'storage.modifier.stan',
        },
      ],
    },
    keywords: {
      patterns: [
        {
          match: '\\b(if|else|for|in|while|return|break|continue)\\b',
          name: 'keyword.control.stan',
        },
        {
          match: '\\b(target|jacobian|print|reject|fatal_error)\\b',
          name: 'keyword.other.stan',
        },
      ],
    },
    constants: {
      patterns: [
        {
          match: '\\b(true|false|positive_infinity|negative_infinity|not_a_number)\\b',
          name: 'constant.language.stan',
        },
      ],
    },
    numbers: {
      patterns: [
        {
          match: '(?<![A-Za-z_])(?:\\d+\\.\\d*|\\.\\d+|\\d+)(?:[eE][+-]?\\d+)?(?![A-Za-z_])',
          name: 'constant.numeric.stan',
        },
      ],
    },
    functions: {
      patterns: [
        {
          match: '\\b[A-Za-z_][A-Za-z0-9_]*(?=\\s*\\()',
          name: 'entity.name.function.stan',
        },
      ],
    },
    operators: {
      patterns: [
        {
          match: '~|\\+=|-=|\\*=|/=|==|!=|<=|>=|&&|\\|\\||[=+\\-*/%^<>?:|]',
          name: 'keyword.operator.stan',
        },
      ],
    },
  },
} satisfies LanguageRegistration
