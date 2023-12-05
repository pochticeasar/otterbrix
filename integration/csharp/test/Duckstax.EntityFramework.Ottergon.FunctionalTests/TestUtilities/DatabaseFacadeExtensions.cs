﻿using Microsoft.EntityFrameworkCore.Infrastructure;

namespace Duckstax.EntityFramework.otterbrix.FunctionalTests.TestUtilities;

public static class DatabaseFacadeExtensions
{
    public static void EnsureClean(this DatabaseFacade databaseFacade)
        => new SampleProviderDatabaseCleaner().Clean(databaseFacade);

}
