/************************************************************************/
/*   crpalign - Chinese Restaurant Process string pair aligner          */
/*   Copyright © 2013 Mans Hulden                                       */
/*                                                                      */
/*   This file is part of crpalign.                                     */
/*                                                                      */
/* Licensed under the Apache License, Version 2.0 (the "License");      */
/* you may not use this file except in compliance with the License.     */
/* You may obtain a copy of the License at                              */
/*                                                                      */
/*     http://www.apache.org/licenses/LICENSE-2.0                       */
/*                                                                      */
/************************************************************************/

/* To build python bindings: gcc -O3 -Wall -Wextra -shared align.c -o libalign.so */
/* WARNING: currently not thread-safe */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdarg.h>
#include <unistd.h>
#include <getopt.h>

/* Yields minimum of three values */
#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))
/* Compares three values, yielding -1, 0, 1 depending on if a, b, or c is the smallest */
#define CMP3(a, b, c) ((a) < (b) ? ((a) < (c) ? (-1) : (1)) : ((b) < (c) ? (0) : (1)))

#define LEFT -1
#define DIAG  0
#define DOWN  1

#define INPUT_FORMAT_L2P  0
#define INPUT_FORMAT_NEWS 1

#define OUTPUT_FORMAT_PLAIN         0
#define OUTPUT_FORMAT_ALIGNED       1
#define OUTPUT_FORMAT_PHONETISAURUS 2
#define OUTPUT_FORMAT_M2M           3

#define MATRIX_MODE_MED   0
#define MATRIX_MODE_GS    1

#define CHAR_SIZE 4096

int g_maxsymbol = 0;
int g_debug = 0;
int g_med = 0;
int g_in_result[CHAR_SIZE];
int g_out_result[CHAR_SIZE];
int g_paircount = 0;
int g_distinct_pairs = 0;
int g_input_format = INPUT_FORMAT_L2P;
int g_output_format = OUTPUT_FORMAT_ALIGNED;
double g_trellis[CHAR_SIZE][CHAR_SIZE];
int g_backptr[CHAR_SIZE][CHAR_SIZE];
int g_current_count[CHAR_SIZE][CHAR_SIZE];
int g_global_count[CHAR_SIZE][CHAR_SIZE];
char *g_symboltable[CHAR_SIZE*4];
double g_prior = 0.1;
double g_zero = 0.0;

struct stringpair {          /* These are all               */
    int *in;                 /* -1 terminated int sequences */
    int *out;
    int *inaligned;
    int *outaligned;
    struct stringpair *next;
} *g_stringpairs = NULL, *g_stringpairs_tail = NULL;

void align_init(void) {
	g_stringpairs = NULL;
	g_stringpairs_tail = NULL;
}

int intseqlen(int *seq) {
    int i;
    for (i = 0; seq[i] != -1; i++) { }
    return i;
}

double log_add(double logy, double logx) {
	/* Supposes that inputs are negative log probabilities */
	if (logy > logx) {
		double temp = logx;
		logx = logy;
		logy = temp;
	}
	double negdiff = logx - logy;
	if (negdiff > 80) {
		return(logy);
	}
	return logx - log(1 + exp(logx - logy));
}

void debug(const char *fmt, ...) {
	va_list arg;
	if (g_debug == 1) {
		va_start(arg, fmt);
		vprintf(fmt, arg);
		va_end(arg);
	}
}

/* Gives length in bytes of UTF-8 character */
int utf8len(char *str) {
	unsigned char s;
	s = (unsigned char)(unsigned int) (*str);
	if (s < 0x80)
		return 1;
	if ((s & 0xe0) == 0xc0) {
		return 2;
	}
	if ((s & 0xf0) == 0xe0) {
		return 3;
	}
	if ((s & 0xf8) == 0xf0) {
		return 4;
	}
	return 0;
}

/* Reverses an integer sequence in-place */
void vector_reverse(int *s, int length) {
	int c, i, j;
	for (i = 0, j = length - 1; i < j; i++, j--) {
		c = s[i];
		s[i] = s[j];
		s[j] = c;
	}
}

/* Returns number of UTF8 characters in char array */
int utf8strlen(char *str) {
	int i,j, len;
	len = strlen(str);
	for (i = 0, j = 0; *(str+i) != '\0' && i < len; j++) {
		i = i + utf8len(str+i);
	}
	return j;
}

int random_3draw(double a, double b, double c) {

    /* From three negative logprobs, do a weighted coin toss */
    /* proportional to each probability, returing -1, 0, 1   */
    /* depending on if a, b, or c is drawn.                  */

	double minv, subv, rand;
	/* Scale neg logprobs */
	minv = MIN3(a, b, c);
	if (minv >= 2) {
		subv = minv - 2; /* <= we scale so that highest prob entry is 2 (in -log space) */
		a -= subv;       /* This to avoid underflow when converting to reals            */
		b -= subv;       /* for the weighted random choice.                             */
		c -= subv;
	}
	a = exp(-a);         /* Convert to three probabilities */
	b = exp(-b);
	c = exp(-c);
	rand = drand48();
	rand = rand * (a + b + c);
	if (rand < a)   { return -1; }
	if (rand < a+b) { return  0; }
	return 1;
}

/* Fills trellis with aligned integer sequences in and out, using the callback function  */
/* cost().  Returns aligned strings in g_in_result[] and g_out_result[]                  */
/* If mode = MODE_GS, we resample alignments by a CRP process (filling trellis "forward" */
/*                    and then drawing a new alignment going "backward")                 */
/* If mode = MODE_MED, we find the "cheapest" alignment                                  */

double fill_trellis(int *in, int *out, double(*cost)(int, int), int mode) {
    int i, x, y, inlen, outlen;
    double left, down, diag, p;
    inlen = intseqlen(in);
    outlen = intseqlen(out);
    g_trellis[0][0] = g_zero;
    for (x = 1; x <= outlen; x++) {
		g_trellis[x][0] = g_trellis[x-1][0] + cost(0,out[x-1]);
		g_backptr[x][0] = LEFT;
    }
    for (y = 1; y <= inlen; y++) {
		g_trellis[0][y] = g_trellis[0][y-1] + cost(in[y-1], 0);
		g_backptr[0][y] = DOWN;
    }
    for (x = 1; x <= outlen; x++) {
		for (y = 1; y <= inlen; y++) {
			left = g_trellis[x-1][y] + cost(0,out[x-1]);
			down = g_trellis[x][y-1] + cost(in[y-1], 0);
			diag = g_trellis[x-1][y-1] + cost(in[y-1], out[x-1]);

			if (mode == MATRIX_MODE_MED) {
				g_trellis[x][y] = MIN3(left, diag, down);
				g_backptr[x][y] = CMP3(left, diag, down);
			}
			else if (mode == MATRIX_MODE_GS) {
				g_trellis[x][y] = log_add(log_add(left, diag), down);
			}
		}
    }

    /* Resample a new "path" for the string pair <in:out> starting from upper right-hand corner
       in the matrix and moving left, down, or diagonally down/left until we reach [0,0]
       ..[B][A]   To choose the direction we do a weighted coin toss between choices A -> B, A -> C, A -> D:
       ..[C][D]   w(B) = p(B) * p(B->A) ; w(C) = p(C) * p(C->A) ; w(D) = p(D) * p(D -> A).
          .  .    and p(X->Y) = the probability of taking the transition (X->Y)
          .  .    Since we've stored the probabilities in log space, we need to do some scaling
                  and conversion before doing the weighted toss.
    */

    if (mode == MATRIX_MODE_GS) {
		for (y = inlen, x = outlen; x > 0 || y > 0 ; ) {
			if (x == 0) {
				y--;
			} else if (y == 0) {
				x--;
			} else {
				left = g_trellis[x-1][y] + cost(0,out[x-1]);
				down = g_trellis[x][y-1] + cost(in[y-1], 0);
				diag = g_trellis[x-1][y-1] + cost(in[y-1], out[x-1]);
				g_backptr[x][y] = random_3draw(left, diag, down);
				x--;
				y--;
			}
		}
    }

    for (i = 0, y = inlen, x = outlen; x > 0 || y > 0; i++) {
		if (g_backptr[x][y] == DIAG) {
			x--;
			y--;
			g_in_result[i] = in[y];
			g_out_result[i] = out[x];
		} else if (g_backptr[x][y] == LEFT) {
			x--;
			g_in_result[i] = 0;
			g_out_result[i] = out[x];
		} else if (g_backptr[x][y] == DOWN) {
			y--;
			g_in_result[i] = in[y];
			g_out_result[i] = 0;
		}
    }

    g_in_result[i] = -1;
    g_out_result[i] = -1;

    vector_reverse(g_in_result, i);
    vector_reverse(g_out_result, i);
    p = g_trellis[outlen][inlen];
    return(p);
}

/* Removes the counts of symbol pairs in two -1 -terminated sequences */
/* to the current count table                                         */
void remove_counts(int *in, int *out) {
	int i;
	for (i = 0; in[i] != -1 && out[i] != -1; i++) {
		g_current_count[in[i]][out[i]]--;
		if (g_current_count[in[i]][out[i]] == 0) {
			g_distinct_pairs--;
		}
	}
}

/* Add the counts of symbol pairs in two -1 -terminated sequences */
/* to the current count table                                     */
void add_counts(int *in, int *out) {
	int i;
	for (i = 0; in[i] != -1 && out[i] != -1; i++) {
		g_current_count[in[i]][out[i]]++;
		g_paircount++;
		if (g_current_count[in[i]][out[i]] == 1) {
			g_distinct_pairs++;
		}
	}
}

/* Add running counts of pairs to the global count table */
void add_global_counts() {
	int i, j;
	for (i = 0; i <= g_maxsymbol; i++) {
		for (j = 0; j <= g_maxsymbol; j++) {
			g_global_count[i][j] += g_current_count[i][j];
		}
	}
}

void print_counts() {
	int i, j;
	debug("\n");
	for (i = 0; i <= g_maxsymbol; i++) {
		for (j = 0; j <= g_maxsymbol; j++) {
			debug("%i ", g_current_count[i][j]);
		}
		debug("\n");
	}
}

/* Cost function called by fill_trellis for MED */
double cost_levenshtein(int a, int b) {
	if (a != b) {
		return 1.0;
	}
	return 0.0;
}

/* Cost function called by fill_trellis for CRP alignment */
double cost_crp(int in, int out) {
    double cost;
    cost = (double)( g_current_count[in][out] + g_prior ) / (double)( g_paircount + g_distinct_pairs * g_prior );
    return(-log(cost));
}

/* Initially, align all string pairs greedily, i.e. e.g. <abcd, ax> => <abcd,ax00> */
void initial_align() {
    struct stringpair *pair;
    int inlen, outlen, i, j, k;
    for (pair = g_stringpairs; pair != NULL; pair = pair->next) {
		inlen = intseqlen(pair->in);
		outlen = intseqlen(pair->out);
		pair->inaligned = malloc(sizeof(int) * (inlen+outlen+1));
		pair->outaligned = malloc(sizeof(int) * (inlen+outlen+1));

		for (i = 0, j = 0, k = 0; pair->in[i] != -1 || pair->out[j] != -1; k++) {
			if (pair->in[i] == -1) {
				pair->inaligned[k] = 0;
				pair->outaligned[k] = pair->out[j];
				j++;
			}
			else if (pair->out[j] == -1) {
				pair->inaligned[k] = pair->in[i];
				pair->outaligned[k] = 0;
				i++;
			} else {
				pair->inaligned[k] = pair->in[i];
				pair->outaligned[k] = pair->out[j];
				i++;
				j++;
			}
		}
		pair->inaligned[k] = -1;
		pair->outaligned[k] = -1;
		add_counts(pair->inaligned, pair->outaligned);
	}
}

/* Align a set of string pairs by minimum edit distance (for reference) */
void med_align() {
    struct stringpair *sp;
    int j;
    for (sp = g_stringpairs; sp != NULL; sp = sp->next) {
		fill_trellis(sp->in, sp->out, &cost_levenshtein, MATRIX_MODE_MED); /* Fill trellis */
		for (j = 0; g_in_result[j] != -1; j++) {
			sp->inaligned[j] = g_in_result[j];
			sp->outaligned[j] = g_out_result[j];
		}
		sp->inaligned[j] = -1;
		sp->outaligned[j] = -1;
    }
}

void crp_align() {
    struct stringpair *sp;
    int j;
    for (sp = g_stringpairs; sp != NULL; sp = sp->next) {
		fill_trellis(sp->in, sp->out, &cost_crp, MATRIX_MODE_MED);
		for (j = 0; g_in_result[j] != -1; j++) {
			sp->inaligned[j] = g_in_result[j];
			sp->outaligned[j] = g_out_result[j];
		}
		sp->inaligned[j] = -1;
		sp->outaligned[j] = -1;
	}
}

void crp_train(int iterations, int burnin, int lag) {
    struct stringpair *sp;
    int i, j;
	for (i = 0; i < iterations; i++) {
		fprintf(stderr,"Alignment iteration: %i\n", i);
		print_counts();
		for (sp = g_stringpairs; sp != NULL; sp = sp->next) {
			remove_counts(sp->inaligned, sp->outaligned);  /* Remove counts before aligning */
			fill_trellis(sp->in, sp->out, &cost_crp, MATRIX_MODE_GS);
			for (j = 0; g_in_result[j] != -1; j++) {
				sp->inaligned[j] = g_in_result[j];
				sp->outaligned[j] = g_out_result[j];
			}
			sp->inaligned[j] = -1;
			sp->outaligned[j] = -1;
			add_counts(sp->inaligned, sp->outaligned);  /* Add counts back from new alignment */
		}
		if (i > burnin && i % lag == 0) {
			add_global_counts();
		}
    }
}

int get_set_char_num(char *utfstring) {
    int i;
    debug("Finding symbol %s with len %i... ", utfstring, utf8len(utfstring));
	for (i = 1; i <= g_maxsymbol; i++) {
		if (strcmp(utfstring, g_symboltable[i]) == 0) {
			debug("Found at %i\n", i);
			return i;
		}
	}
	g_maxsymbol++;
	debug("Not found, adding at %i\n", g_maxsymbol);
	g_symboltable[g_maxsymbol] = strdup(utfstring);
	return(g_maxsymbol);
}

/* Reads character sequences in and out and onverts them to integer sequences */
/* And adds them to the global list of integer sequence pairs                 */

void add_string_pair(char *in, char *out) {
    int *int_in, *int_out;
    int i, j;
    char *token;
    struct stringpair *newpair;
    /* Get int array */
    int_in  = malloc(sizeof(int) * (utf8strlen(in) + 1));
    int_out = malloc(sizeof(int) * (utf8strlen(out) + 1));
	if (g_input_format == INPUT_FORMAT_L2P) {
		for (i = 0, j = 0; in[i] != '\0'; i += utf8len(&in[i]), j++) {
			int_in[j] = get_set_char_num(strndup(&in[i], utf8len(&in[i])));
		}
		int_in[j] = -1;
		for (i = 0, j = 0; out[i] != '\0'; i += utf8len(&out[i]), j++) {
			int_out[j] = get_set_char_num(strndup(&out[i], utf8len(&out[i])));
		}
		int_out[j] = -1;
	} else if (g_input_format == INPUT_FORMAT_NEWS) {
		token = strtok(in, " ");
		for (j = 0; token != NULL; j++) {
			int_in[j] = get_set_char_num(token);
			token = strtok(NULL, " ");
		}
		int_in[j] = -1;
		token = strtok(out, " ");
		for (j = 0; token != NULL; j++) {
			int_out[j] = get_set_char_num(token);
			token = strtok(NULL, " ");
		}
		int_out[j] = -1;
	}

	newpair = malloc(sizeof(struct stringpair));
	newpair->in = int_in;
	newpair->out = int_out;
	newpair->next = NULL;
	if (g_stringpairs == NULL) {
		g_stringpairs = newpair;
		g_stringpairs_tail = newpair;
	} else {
		g_stringpairs_tail->next = newpair;
		g_stringpairs_tail = newpair;
	}
}

/* Directly add two -1 terminated integer sequences */
void add_int_pair(int *in, int *out) {
	int inlen, outlen;
    struct stringpair *newpair;
    newpair = malloc(sizeof(struct stringpair));
	inlen = intseqlen(in) + 1;
	outlen = intseqlen(out) + 1;
	newpair->in = malloc(inlen * sizeof(int));
	newpair->out = malloc(outlen * sizeof(int));
	memcpy(newpair->in, in, inlen * sizeof(int));
	memcpy(newpair->out, out, outlen * sizeof(int));
    newpair->next = NULL;
    if (g_stringpairs == NULL) {
		g_stringpairs = newpair;
		g_stringpairs_tail = newpair;
    } else {
		g_stringpairs_tail->next = newpair;
		g_stringpairs_tail = newpair;
    }
}

void clear_counts() {
	int i,j;
	for (i = 0; i <= g_maxsymbol; i++) {
		for (j = 0; j <= g_maxsymbol; j++) {
			g_current_count[i][j] = 0;
			g_global_count[i][j] = 0;
		}
	}
}

void print_pair_plain(int *in, int *out) {
	int i;
	g_symboltable[0] = " ";
	for (i = 0; in[i] != -1; i++) {
		printf("%s", in[i] == 0 ? " " : g_symboltable[ in[i] ]);
	}
	printf("\n");
	for (i = 0; out[i] != -1; i++) {
		printf("%s", out[i] == 0 ? " " : g_symboltable[ out[i] ]);
	}
	printf("\n\n");
}

void print_pair_m2m(int *in, int *out) {
	int i;
	g_symboltable[0] = "_";
	for (i = 0; in[i] != -1; i++) {
		printf("%s|", in[i] == 0 ? " " : g_symboltable[ in[i] ]);
	}
	printf("\t");
	for (i = 0; out[i] != -1; i++) {
		printf("%s|", out[i] == 0 ? " " : g_symboltable[ out[i] ]);
	}
	printf("\n");
}

void print_pair_phonetisaurus(int *in, int *out) {
	int i;
	g_symboltable[0] = "_";
	for (i = 0; in[i] != -1 && out[i] != -1; i++) {
		printf("%s}%s", g_symboltable[in[i]], g_symboltable[out[i]]);
		if (in[i+1] != -1 && out[i+1] != -1) {
			printf(" ");
		}
	}
	printf("\n");
}

void print_pair_aligned(int *in, int *out) {
	int i, fieldwidth;
	char *instr, *outstr;
	g_symboltable[0] = "_";
	for (i = 0; in[i] != -1 && out[i] != -1; i++) {
		instr = g_symboltable[ in[i] ];
		outstr =  g_symboltable[ out[i] ];
		fieldwidth = utf8strlen(instr) > utf8strlen(outstr) ? utf8strlen(instr) : utf8strlen(outstr);
		printf("%-*s", fieldwidth, instr);
		if (in[i+1] != -1 && out[i+1] != -1)
			printf("|");
	}
	printf("\n");
	for (i = 0; in[i] != -1 && out[i] != -1; i++) {
		instr = g_symboltable[ in[i] ];
		outstr =  g_symboltable[ out[i] ];
		fieldwidth = utf8strlen(instr) > utf8strlen(outstr) ? utf8strlen(instr) : utf8strlen(outstr);
		printf("%-*s", fieldwidth, outstr);
		if (in[i+1] != -1 && out[i+1] != -1)
			printf("|");
	}
	printf("\n\n");
}

/* Functions for Python ctypes wrap */

struct stringpair *getpairs_init() {
	return g_stringpairs;
}

int *getpairs_in(struct stringpair *sp) {
	return sp->inaligned;
}

int *getpairs_out(struct stringpair *sp) {
	return sp->outaligned;
}

struct stringpair *getpairs_advance(struct stringpair *sp) {
	return sp->next;
}

/************************************/

void write_stringpairs() {
	struct stringpair *sp;
	for (sp = g_stringpairs; sp != NULL; sp = sp->next) {
		switch(g_output_format) {
			case OUTPUT_FORMAT_PLAIN:
			print_pair_plain(sp->inaligned, sp->outaligned);
			break;
			case OUTPUT_FORMAT_ALIGNED:
			print_pair_aligned(sp->inaligned, sp->outaligned);
			break;
			case OUTPUT_FORMAT_PHONETISAURUS:
			print_pair_phonetisaurus(sp->inaligned, sp->outaligned);
			break;
			case OUTPUT_FORMAT_M2M:
			print_pair_m2m(sp->inaligned, sp->outaligned);
			break;
		}
	}
}

void read_stringpairs() {
	char *my_string = NULL, *token1, *token2;
	char str1[1024], str2[1024];
	size_t nbytes;
	int bytes_read;
	while ((bytes_read = getline(&my_string, &nbytes, stdin)) != -1) {
		if (g_input_format == INPUT_FORMAT_L2P) {
			if (sscanf(my_string, "%1023s %1023s", &str1[0], &str2[0]) == 2)
				add_string_pair(str1, str2);
		} else if (g_input_format == INPUT_FORMAT_NEWS) {
			token1 = strtok(my_string, "\t\n");
			token2 = strtok(NULL, "\t\n");
			if (token1 != NULL && token2 != NULL)
				add_string_pair(token1, token2);
		}
	}
	clear_counts();
	initial_align();
}

int main(int argc, char **argv) {
	static char *usagestring =
	"Chinese restaurant process string pair aligner\n"
	"Basic usage: crpalign11 [options] < infile.txt > aligned.txt\n"
	"             infile.txt is a list of TAB-separated word-pairs, one pair per line.\n\n"
	"Options:\n"
	"-d     --debug           print debug info\n"
	"-h     --help            help\n"
	"-m     --med             do simple med-alignment only (for comparison)\n"
	"-x NUM --iterations=NUM  run aligner for NUM iterations (default 10)\n"
	"-i FMT --informat=FMT    expect data in format FMT=l2p|news (default l2p)\n"
	"-o FMT --outformat=FMT   print data in format FMT=plain|aligned|phonetisaurus|m2m\n"
	"-b NUM --burnin=NUM      run Gibbs sampler with NUM iterations of burn-in\n"
	"-l NUM --lag=NUM         collect counts from sampler every NUM iterations\n"
	"-p NUM --prior=NUM       use a prior of NUM for sampler (default 0.1)\n";


	int opt, iterations = 10, burnin = 5, lag = 1, option_index = 0;
	static struct option long_options[] =
	{
		{"debug",       no_argument,       0, 'd'},
		{"help",        no_argument,       0, 'h'},
		{"med",         no_argument,       0, 'm'},
		{"iterations",  required_argument, 0, 'x'},
		{"informat",    required_argument, 0, 'i'},
		{"outformat",   required_argument, 0, 'o'},
		{"burnin",      required_argument, 0, 'b'},
		{"lag",         required_argument, 0, 'l'},
		{"prior",       required_argument, 0, 'p'},
		{0, 0, 0, 0}
	};

	while ((opt = getopt_long(argc, argv, "dmx:b:l:p:i:o:h", long_options, &option_index)) != -1) {
		switch(opt) {
			case 'd':
			g_debug = 1;
			break;
			case 'm':
			g_med = 1;
			break;
			case 'x':
			iterations = atoi(optarg);
			break;
			case 'b':
			burnin = atoi(optarg);
			break;
			case 'h':
			printf("%s", usagestring);
			exit(0);
			case 'i':
			if (strcmp(optarg,"l2p") == 0) {
				g_input_format = INPUT_FORMAT_L2P;
			} else if (strcmp(optarg, "news") == 0) {
				g_input_format = INPUT_FORMAT_NEWS;
			} else {
				fprintf(stderr, "Invalid option %s for input format\n", optarg);
				exit(EXIT_FAILURE);
			}
			break;
			case 'o':
			if (strcmp(optarg,"plain") == 0) {
				g_output_format = OUTPUT_FORMAT_PLAIN;
			} else if (strcmp(optarg, "aligned") == 0) {
				g_output_format = OUTPUT_FORMAT_ALIGNED;
			} else if (strcmp(optarg, "phonetisaurus") == 0) {
				g_output_format = OUTPUT_FORMAT_PHONETISAURUS;
			} else if (strcmp(optarg, "m2m") == 0) {
				g_output_format = OUTPUT_FORMAT_M2M;
			} else {
				fprintf(stderr, "Invalid option %s for output format\n", optarg);
				exit(EXIT_FAILURE);
			}
			break;
			case 'l':
			lag = atoi(optarg);
			break;
			case 'p':
			g_prior = strtod(optarg,NULL);
			break;
		}
	}

	srand48((unsigned int)time((time_t *)NULL));
	read_stringpairs();
	if (g_med == 1) {
		med_align();
	} else {
		crp_train(iterations,burnin,lag);
		crp_align();
	}
	write_stringpairs();
	return(0);
}
