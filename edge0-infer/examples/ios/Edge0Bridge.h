#import <Foundation/Foundation.h>
NS_ASSUME_NONNULL_BEGIN
// Lifecycle and generation use one worker. Queries use a separate queue and a lifecycle lock.
@interface Edge0Bridge : NSObject
- (NSArray<NSDictionary<NSString *, id> *> *)modelsAt:(NSString *)root;
- (NSDictionary<NSString *, id> *)load:(NSString *)path;
- (NSDictionary<NSString *, id> *)generate:(NSArray<NSDictionary<NSString *, NSString *> *> *)messages
                               thinking:(BOOL)thinking
                            temperature:(double)temperature
                              maxTokens:(nullable NSNumber *)maxTokens
                            tokenEvents:(BOOL)tokenEvents
                                receive:(void (^)(NSString *kind, NSString *text))receive;
- (NSDictionary<NSString *, id> *)progress;
- (NSDictionary<NSString *, id> *)snapshot;
- (NSDictionary<NSString *, id> *)systemStatus;
- (NSDictionary<NSString *, id> *)reset;
- (void)prepareGeneration;
- (void)cancel;
- (NSDictionary<NSString *, id> *)close;
@end
NS_ASSUME_NONNULL_END
